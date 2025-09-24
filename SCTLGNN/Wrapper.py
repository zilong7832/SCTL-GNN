import os
import math
import copy
from copy import deepcopy
from typing import Iterable, Optional, Dict

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def build_pairs(
    Z_torch: torch.Tensor,          # [N,H] 你的 cell embedding，在哪个设备都行
    k: int = 10,
    tau: float = 0.7,
    symmetrize: bool = True,
    index_map: np.ndarray | None = None,   # 本地 -> 全局 索引映射（例如 split["train"]）
) -> torch.Tensor:
    """
    在 CPU 上用 KNN(余弦)构建锚点边：
      - 只取每个样本的 top-k 邻居（避免 O(N^2)）
      - 用 softmax(-dist/tau) 得到权重
      - 返回 CPU tensor: [M,3] = [i_global, j_global, w_ij]
    """
    if Z_torch.ndim != 2 or Z_torch.size(0) <= 1:
        return torch.empty(0, 3, dtype=torch.float32)

    # 拷到 CPU float32
    Z = Z_torch.detach().to("cpu", dtype=torch.float32).numpy()
    # 行归一化：cosine 距离 = 1 - cos_sim
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    n = Z.shape[0]
    n_neighbors = min(k + 1, n)  # 包含 self，后面会去掉
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    nn.fit(Z)
    dists, idxs = nn.kneighbors(Z, return_distance=True)  # [N,k+1]

    # 去掉 self（通常第一列）
    idxs = idxs[:, 1:]                 # [N,k]
    dists = dists[:, 1:]               # [N,k]

    # 按行 softmax(-dist/tau) 得权重
    S = np.exp(-dists / max(tau, 1e-12))
    S /= (S.sum(axis=1, keepdims=True) + 1e-12)

    rows = np.arange(n, dtype=np.int64)[:, None]
    i_local = np.repeat(rows, idxs.shape[1], axis=1).reshape(-1)  # [N*k]
    j_local = idxs.reshape(-1).astype(np.int64)                   # [N*k]
    w_flat  = S.reshape(-1).astype(np.float32)                    # [N*k]

    # 本地 -> 全局
    if index_map is not None:
        i_glob = index_map[i_local]
        j_glob = index_map[j_local]
    else:
        i_glob = i_local
        j_glob = j_local

    if symmetrize:
        i2 = np.concatenate([i_glob, j_glob], axis=0)
        j2 = np.concatenate([j_glob, i_glob], axis=0)
        w2 = np.concatenate([w_flat,  w_flat], axis=0) * 0.5
        out = np.stack([i2, j2, w2], axis=1)
    else:
        out = np.stack([i_glob, j_glob, w_flat], axis=1)

    return torch.from_numpy(out)  # 先留在 CPU；用 loss 时再 .to(device)

def anchor_smoothing_loss(
    y_pred: torch.Tensor,        # [N, P]
    anchor_pairs: torch.Tensor,  # [M, 3] = [i, j, w_ij] （device 无所谓）
    normalize: bool = True,
    eps: float = 1e-8
) -> torch.Tensor:
    """ L_anchor = sum w_ij * ||y_i - y_j||^2  / sum w_ij """
    if anchor_pairs is None or anchor_pairs.numel() == 0:
        return y_pred.new_tensor(0.0)
    i = anchor_pairs[:, 0].long()
    j = anchor_pairs[:, 1].long()
    w = anchor_pairs[:, 2].float()
    diff = (y_pred[i] - y_pred[j]).pow(2).mean(dim=1)  # [M]
    num = (w * diff).sum()
    if normalize:
        den = w.sum().clamp_min(eps)
        return num / den
    return num / (len(w) + eps)

def filter_anchor_pairs_by_index(anchor_pairs: torch.Tensor, allowed_idx: torch.Tensor) -> torch.Tensor:
    """可选：仅保留两端都在 allowed_idx 的边。"""
    if anchor_pairs is None or anchor_pairs.numel() == 0:
        return anchor_pairs
    mask_i = torch.isin(anchor_pairs[:, 0].long(), allowed_idx.cpu())
    mask_j = torch.isin(anchor_pairs[:, 1].long(), allowed_idx.cpu())
    keep = mask_i & mask_j
    return anchor_pairs[keep]

def calculate_weight_map(obs_celltypes, target_ct, similarity_matrix, sample_counts, beta, input_dim, device):
    """
    This function contains the core logic from V1 to calculate weights based on normalized inverse risk.
    It is called by the new function when the sample size is less than 1000.
    It calculates weights ONLY for the cell types provided in the `obs_celltypes` list.
    """
    related_cts = list({ct for ct in obs_celltypes if ct != target_ct})
    
    if target_ct not in sample_counts or sample_counts[target_ct] == 0:
        return {} # Return empty weights if target has no samples

    n_Q = sample_counts[target_ct]
    d = input_dim if input_dim is not None else 50
    
    # 1. Calculate risk for the target domain (r_Q)
    r_Q = n_Q ** (-2 * beta / (2 * beta + d))

    # 2. Calculate risk for each related source domain (r_Pj)
    r_P = {}
    for ct in related_cts:
        if ct not in sample_counts or sample_counts[ct] == 0:
            continue
        n_P = sample_counts[ct]
        gamma = similarity_matrix.get(target_ct, {}).get(ct, 0) # Safely get similarity score
        
        exponent = -2 * gamma * beta / (2 * gamma * beta + d)
        r_pj = n_P ** exponent
        r_P[ct] = r_pj
    
    # 3. Compute weights as normalized inverse-risk
    inv_r_Q = 1.0 / r_Q
    inv_r_P = {ct: 1.0 / r for ct, r in r_P.items()}
    denom = inv_r_Q + sum(inv_r_P.values())
    
    # Create a map of cell type to its calculated weight
    calculated_weights = {}
    if denom > 0:
        calculated_weights[target_ct] = inv_r_Q / denom
        for ct in related_cts:
            if ct in inv_r_P:
                calculated_weights[ct] = inv_r_P[ct] / denom
    else: # Fallback in case of issues
        calculated_weights[target_ct] = 1.0
        
    return calculated_weights

# New main function that implements your hypothesis
def compute_weights(obs_celltypes, target_ct, related_cts, similarity_matrix, sample_counts, beta, input_dim, device):
    """
    Computes sample-wise weights based on a hypothesis about the target cell type's sample size.

    - If sample size >= 1000: Use focused training. Target weight is 1, all others are 0.
    - If sample size < 1000: Use weighted transfer learning. Calculate weights for target and related 
      cell types using inverse-risk, all others are 0.
    """
    n_Q = sample_counts.get(target_ct, 0)
    
    weights_map = {}
    
    # Case 1: Sample size is large enough --> Focused Training
    if n_Q >= 1000:
        print(f"INFO: Target '{target_ct}' has {n_Q} samples (>= 1000). Using focused training (weight=1).")
        weights_map[target_ct] = 1.0
        
    # Case 2: Sample size is small --> Weighted Transfer Learning
    else:
        print(f"INFO: Target '{target_ct}' has {n_Q} samples (< 1000). Using weighted transfer learning.")
        # Define the set of cell types that will have non-zero weights
        active_celltypes = [target_ct] + related_cts
        
        # Calculate weights only for this active subset using the original V1 logic
        weights_map = calculate_weight_map(
            obs_celltypes=active_celltypes,
            target_ct=target_ct,
            similarity_matrix=similarity_matrix,
            sample_counts=sample_counts,
            beta=beta,
            input_dim=input_dim,
            device=device
        )

    # Build the final full-length tensor of weights for all samples in the batch
    final_weights = []
    for ct in obs_celltypes:
        # If a cell type is in our map, use its weight. Otherwise, its weight is 0.
        final_weights.append(weights_map.get(ct, 0.0)) 
        
    w = torch.tensor(final_weights, dtype=torch.float32, device=device)
    return w

class ScTlGNNWrapper:
    """
    Wrapper for ScTlGNN training / inference.

    Notes
    -----
    - All method signatures keep backward compatibility with your previous usage.
    - Device management is made safer in `predict` and `score`.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = ScTlGNN(args).to(args.device)
        self.qt_pred = None
        self.qt_true = None

    # ---------- model I/O ----------
    def save_model(self, dataset, repi, fold, celltype: Optional[str] = None):
        """Save model state_dict with the new naming convention."""
        fname = f"{dataset}_{repi}_{fold}"
        if celltype is not None:
            fname += f"_{celltype}"
        fname += ".pth" 
        
        path = os.path.join(self.args.model_folder, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, dataset, repi, fold, celltype: Optional[str] = None):
        """Load model state_dict using the new naming convention."""
        fname = f"{dataset}_{repi}_{fold}"
        if celltype is not None:
            fname += f"_{celltype}"
        fname += ".pth" 
        
        path = os.path.join(self.args.model_folder, fname)
        state = torch.load(path, map_location=self.args.device)
        self.model.load_state_dict(state)

    # ---------- inference ----------
    def predict(
        self,
        graph: dgl.DGLGraph,
        idx: Optional[Iterable[int]] = None,
        device: str = 'cpu',
        embedding: bool = False
    ) -> torch.Tensor:
        """Forward pass on graph and return predictions on the requested device."""
        src_dev = self.args.device
        model = self.model

        if src_dev != device:
            # Use a shallow copy to avoid mutating the training model's device
            model = copy.deepcopy(self.model).to(device)
            graph = graph.to(device)
        else:
            graph = graph.to(src_dev)
            model = model.to(src_dev)

        model.eval()
        with torch.no_grad():
            out = model.forward(graph, embedding=embedding)
            if idx is not None:
                out = out[idx]
        return out.to(device)

    def score(self, g: dgl.DGLGraph, idx: Iterable[int], labels: torch.Tensor, device: str = 'cpu') -> float:
        """Return RMSE on the given indices."""
        with torch.no_grad():
            preds = F.relu(self.predict(g, idx, device))
            loss = F.mse_loss(preds, labels.to(device)).item()
        return math.sqrt(loss)

    # ---------- training ----------
    def fit(
        self,
        g: dgl.DGLGraph,
        y: torch.Tensor,
        split: Optional[Dict[str, np.ndarray]] = None,
        eval: bool = True,
        verbose: int = 2,
        y_test: Optional[torch.Tensor] = None,
        logger=None,
        sampling: bool = False,
        eval_interval: int = 1,
        dataset: Optional[str] = None,
        repi: Optional[int] = None,
        fold: Optional[int] = None
    ):
        """
        Train the model with optional evaluation and early stopping.
        """
        if sampling:
            # If you have a sampling path implemented elsewhere
            return self.fit_with_sampling(g, y, split, eval, verbose, y_test, logger)

        # Use self.args directly for all parameters
        PREFIX = self.args.prefix
        CELL_SIZE = self.args.CELL_SIZE
        TRAIN_SIZE = self.args.TRAIN_SIZE

        g = g.to(self.args.device)
        y = y.float().to(self.args.device)
        y_test = y_test.float().to(self.args.device) if y_test is not None else None

        if verbose > 1 and logger is None:
            os.makedirs(self.args.log_folder, exist_ok=True)
            # Assuming log filename logic is in the main script
            logger = open(f'{self.args.log_folder}/{PREFIX}_temp.log', 'w')
        if verbose > 1:
            logger.write(str(self.model) + '\n'); logger.flush()

        opt = torch.optim.AdamW(self.model.parameters(),
                                lr=self.args.learning_rate,
                                weight_decay=self.args.weight_decay)
        criterion = nn.MSELoss()

        tr, val, te = [], [], []
        best_val = float('inf')
        best_ep = -1
        best_dict = deepcopy(self.model.state_dict())

        for epoch in range(self.args.epoch):
            if verbose > 1:
                logger.write(f'epoch:  {epoch}\n'); logger.flush()

            self.model.train()
            logits = self.model(g)
            loss = criterion(logits[split['train']], y[split['train']])
            running_loss = loss.item()

            opt.zero_grad()
            loss.backward()

            grad_clip = getattr(self.args, "grad_clip", 0.0)
            if isinstance(grad_clip, (int, float)) and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            opt.step()
            torch.cuda.empty_cache()
            tr.append(math.sqrt(running_loss))

            if epoch % eval_interval == 0:
                val_rmse = self.score(g, split['valid'], y[split['valid']], self.args.device)
                val.append(val_rmse)
                if verbose > 1:
                    logger.write(f'training loss:  {tr[-1]}\n'); logger.flush()
                    logger.write(f'validation loss:  {val[-1]}\n'); logger.flush()

                if eval and (y_test is not None):
                    test_rmse = self.score(g, np.arange(TRAIN_SIZE, CELL_SIZE), y_test, self.args.device)
                    te.append(test_rmse)
                    if verbose > 1:
                        logger.write(f'testing loss:  {te[-1]}\n'); logger.flush()

                if val[-1] < best_val:
                    best_val = val[-1]
                    best_ep = epoch // eval_interval
                    if self.args.save_best:
                        print(f"New best validation score: {best_val:.4f}. Saving model.")
                        self.save_model(dataset, repi, fold)
                    best_dict = deepcopy(self.model.state_dict())

                if epoch > 1500 and self.args.early_stopping > 0 and min(val[-self.args.early_stopping:]) > best_val:
                    if verbose > 1:
                        logger.write('Early stopped.\n'); logger.flush()
                    break

                if epoch > 1200 and epoch % 15 == 0:
                    for p in opt.param_groups:
                        p['lr'] *= self.args.lr_decay

                if verbose > 0:
                    print('epoch', epoch)
                    print('training: ', tr[-1])
                    print('valid: ', val[-1])
                    if eval and (y_test is not None) and len(te) > 0:
                        print('testing: ', te[-1])

        if self.args.save_final:
            state = {'model': self.model, 'optimizer': opt.state_dict(), 'epoch': epoch - 1}
            os.makedirs(self.args.model_folder, exist_ok=True)
            torch.save(state, f'{self.args.model_folder}/{PREFIX}.epoch{epoch}.pth')

        if verbose > 1:
            if eval and (y_test is not None) and len(te) > 0:
                logger.write(f'epoch {best_ep} minimal val {best_val} with training: {tr[best_ep]} and testing: {te[best_ep]}\n')
            else:
                logger.write(f'epoch {best_ep} minimal val {best_val} with training: {tr[best_ep]}\n')

        if verbose > 0 and eval and (y_test is not None) and len(te) > 0:
            print('min testing', min(te), te.index(min(te)))
            print('converged testing', best_ep * eval_interval, te[best_ep])

        self.model.load_state_dict(best_dict)
        return self.model
    
    def fine_tune(
        self,
        g: dgl.DGLGraph,
        y: torch.Tensor,
        split: Dict[str, np.ndarray],
        celltype: str,
        sample_weights: torch.Tensor,
        ft_epochs: int,
        ft_lr: float,
        ft_patience: int,
        verbose: int = 2,
        logger=None,
        eval_interval: int = 1
    ):
        """
        Fine-tunes the model using a pre-computed weight tensor.
        The total loss is a combination of weighted MSE and an optional anchor smoothing regularizer.
        Total Loss = weighted_MSE + lambda_anchor * L_anchor
        """
        device = self.args.device
        g = g.to(device)
        y = y.to(device)

        # Freeze embedding layers for stability during fine-tuning
        for name, param in self.model.named_parameters():
            if name.startswith("embed_feat") or name.startswith("embed_cell"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        opt = torch.optim.AdamW(trainable_params, lr=ft_lr, weight_decay=self.args.weight_decay)

        best_val_loss, epochs_no_improve = float('inf'), 0
        best_ft_dict = deepcopy(self.model.state_dict())

        # --- Anchor Smoothing Loss Setup ---
        lambda_anchor = float(getattr(self.args, "lambda_anchor", 0.0))
        anchor_pairs_full = getattr(self, "anchor_pairs_tensor", None)
        
        if lambda_anchor > 0.0 and anchor_pairs_full is not None and anchor_pairs_full.numel() > 0:
            # Anchor pairs are built only on training data, so we can use them directly.
            # Move them to the correct device once for the fine-tuning loop.
            anchor_pairs_train = anchor_pairs_full.to(device)
        else:
            anchor_pairs_train = None

        train_idx = split["train"]
        
        # --- Fine-tuning Loop ---
        for epoch in range(ft_epochs):
            self.model.train()
            out = self.model(g)  # Full forward pass, returns [N, P] predictions

            # 1) Weighted MSE Loss (Primary Loss)
            losses_per_elem = F.mse_loss(out[train_idx], y[train_idx], reduction="none") # Shape: [N_train, P]
            w_train = sample_weights[train_idx].to(device).unsqueeze(1)                   # Shape: [N_train, 1]
            weighted_mse = (losses_per_elem * w_train).mean()

            total_loss = weighted_mse

            # 2) Anchor Smoothing Loss (Regularizer)
            loss_anchor = torch.tensor(0.0) # Default value
            if anchor_pairs_train is not None:
                # The loss function operates on the full output 'out' using global indices from anchor pairs
                loss_anchor = anchor_smoothing_loss(out, anchor_pairs_train, normalize=True)
                total_loss = total_loss + lambda_anchor * loss_anchor

            opt.zero_grad()
            total_loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            opt.step()

            torch.cuda.empty_cache()

            # --- Validation and Early Stopping ---
            if epoch % eval_interval == 0:
                val_loss = self.score(g, split["valid"], y[split["valid"]], device)

                if verbose:
                    log_msg = (
                        f"[Fine-tune: {celltype}] Epoch {epoch+1:03d}: "
                        f"Train MSE_w={weighted_mse.item():.4f}, "
                        f"Anchor={loss_anchor.item():.4f}, "
                        f"TotalLoss={total_loss.item():.4f} | "
                        f"Val RMSE={val_loss:.4f}"
                    )
                    print(log_msg)
                    if logger:
                        logger.write(log_msg + "\n")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_ft_dict = deepcopy(self.model.state_dict())
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= ft_patience:
                        print(
                            f"Early stopping at epoch {epoch+1} for cell type '{celltype}'. "
                            f"Best val loss: {best_val_loss:.4f}"
                        )
                        break
        
        print(f"Finished fine-tuning for {celltype}. Loading best model with val loss: {best_val_loss:.4f}")
        self.model.load_state_dict(best_ft_dict)

        # Unfreeze all parameters after fine-tuning for this cell type is complete
        for param in self.model.parameters():
            param.requires_grad = True
            
        return self
        
    # def fine_tune(
    #         self,
    #         g: dgl.DGLGraph,
    #         y: torch.Tensor,
    #         split: Dict[str, np.ndarray],
    #         celltype: str,
    #         sample_weights: torch.Tensor,
    #         ft_epochs: int,
    #         ft_lr: float,
    #         ft_patience: int,
    #         verbose: int = 2,
    #         logger=None,
    #         eval_interval: int = 1
    #     ):
    #         """
    #         Fine-tunes with weighted MSE + anchor smoothing (train-only anchors from CPU).
    #         total_loss = weighted_MSE + lambda_anchor * L_anchor
    #         """
    #         device = self.args.device
    #         g = g.to(device)
    #         y = y.to(device)

    #         # freeze as before
    #         for name, param in self.model.named_parameters():
    #             if name.startswith("embed_feat") or name.startswith("embed_cell"):
    #                 param.requires_grad = False
    #             else:
    #                 param.requires_grad = True

    #         trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
    #         opt = torch.optim.AdamW(trainable_params, lr=ft_lr, weight_decay=self.args.weight_decay)

    #         best_val_loss, epochs_no_improve = float('inf'), 0
    #         best_ft_dict = deepcopy(self.model.state_dict())

    #         lambda_anchor = float(getattr(self.args, "lambda_anchor", 0.0))
    #         # 注意：我们已经在 pipeline 里只对 train 构过图并注入 wrapper 了
    #         anchor_pairs_full = getattr(self, "anchor_pairs_tensor", None)
    #         if lambda_anchor > 0.0 and anchor_pairs_full is not None and anchor_pairs_full.numel() > 0:
    #             # 这批边本来就只含 train 节点（因为我们用 index_map=train_idx_np 构的）
    #             anchor_pairs_train = anchor_pairs_full.to(device)
    #         else:
    #             anchor_pairs_train = None  # 不做 anchor 正则

    #         train_idx = split["train"]

    #         for epoch in range(ft_epochs):
    #             self.model.train()
    #             out = self.model(g)  # [N,P]

    #             # 1) weighted MSE（原有）
    #             losses_per_elem = F.mse_loss(out[train_idx], y[train_idx], reduction="none")  # [N_train,P]
    #             w_train = sample_weights[train_idx].to(device).unsqueeze(1)                   # [N_train,1]
    #             weighted_mse = (losses_per_elem * w_train).mean()

    #             total_loss = weighted_mse

    #             # 2) anchor smoothing（若启用）
    #             if lambda_anchor > 0.0 and anchor_pairs_train is not None and anchor_pairs_train.numel() > 0:
    #                 loss_anchor = anchor_smoothing_loss(out, anchor_pairs_train, normalize=True)
    #                 total_loss = total_loss + lambda_anchor * loss_anchor
    #             else:
    #                 loss_anchor = None

    #             opt.zero_grad()
    #             total_loss.backward()
    #             if self.args.grad_clip > 0:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
    #             opt.step()

    #             torch.cuda.empty_cache()

    #             # 验证 RMSE（不变）
    #             val_loss = self.score(g, split["valid"], y[split["valid"]], device)

    #             if verbose and epoch % eval_interval == 0:
    #                 log_msg = f"[Fine-tune: {celltype}] Epoch {epoch+1:03d}: " \
    #                         f"Train MSE_w={weighted_mse.item():.4f}, " \
    #                         f"{('Anchor=' + f'{loss_anchor.item():.4f}, ') if loss_anchor is not None else ''}" \
    #                         f"Val RMSE={val_loss:.4f}"
    #                 print(log_msg)
    #                 if logger:
    #                     logger.write(log_msg + "\n")

    #             # 早停（不变）
    #             if val_loss < best_val_loss:
    #                 best_val_loss = val_loss
    #                 epochs_no_improve = 0
    #                 best_ft_dict = deepcopy(self.model.state_dict())
    #             else:
    #                 epochs_no_improve += 1
    #                 if epochs_no_improve >= ft_patience:
    #                     print(f"Early stopping at epoch {epoch+1} for cell type '{celltype}'. "
    #                         f"Best val loss: {best_val_loss:.4f}")
    #                     break

    #         print(f"Finished fine-tuning for {celltype}. Loading best model with val loss: {best_val_loss:.4f}")
    #         self.model.load_state_dict(best_ft_dict)

    #         # unfreeze
    #         for param in self.model.parameters():
    #             param.requires_grad = True
    #         return self


class ScTlGNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.nrc = args.no_readout_concatenate

        hid_feats = args.hidden_size
        out_feats = args.OUTPUT_SIZE
        FEATURE_SIZE = args.FEATURE_SIZE

        # Optional batch features encoder
        if not args.no_batch_features:
            self.extra_encoder = nn.Linear(args.BATCH_NUM, hid_feats)

        # Cell embedding
        if args.cell_init == 'none':
            self.embed_cell = nn.Embedding(args.CELL_SIZE, hid_feats)
        else:
            # e.g., SVD-100 used as cell features
            self.embed_cell = nn.Linear(100, hid_feats)

        # Feature (gene) embedding
        self.embed_feat = nn.Embedding(FEATURE_SIZE, hid_feats)

        # Input towers (feature & cell)
        self.input_linears = nn.ModuleList()
        self.input_acts = nn.ModuleList()
        self.input_norm = nn.ModuleList()
        for i in range((args.embedding_layers - 1) * 2):
            self.input_linears.append(nn.Linear(hid_feats, hid_feats))
        if args.activation == 'gelu':
            self.input_acts += [nn.GELU() for _ in range((args.embedding_layers - 1) * 2)]
        elif args.activation == 'prelu':
            self.input_acts += [nn.PReLU() for _ in range((args.embedding_layers - 1) * 2)]
        elif args.activation == 'relu':
            self.input_acts += [nn.ReLU() for _ in range((args.embedding_layers - 1) * 2)]
        elif args.activation == 'leaky_relu':
            self.input_acts += [nn.LeakyReLU() for _ in range((args.embedding_layers - 1) * 2)]

        if args.normalization == 'batch':
            self.input_norm += [nn.BatchNorm1d(hid_feats) for _ in range((args.embedding_layers - 1) * 2)]
        elif args.normalization == 'layer':
            self.input_norm += [nn.LayerNorm(hid_feats) for _ in range((args.embedding_layers - 1) * 2)]
        elif args.normalization == 'group':
            self.input_norm += [nn.GroupNorm(4, hid_feats) for _ in range((args.embedding_layers - 1) * 2)]

        self.edges = ['feature2cell', 'cell2feature']

        # Hetero SAGE conv stack
        self.conv_layers = nn.ModuleList()
        if args.residual == 'res_cat':
            self.conv_layers.append(
                dglnn.HeteroGraphConv(
                    dict(zip(self.edges, [
                        dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats,
                                       aggregator_type=args.agg_function, norm=None)
                        for _ in range(len(self.edges))
                    ])),
                    aggregate='stack'
                )
            )
            for _ in range(args.conv_layers - 1):
                self.conv_layers.append(
                    dglnn.HeteroGraphConv(
                        dict(zip(self.edges, [
                            dglnn.SAGEConv(in_feats=hid_feats * 2, out_feats=hid_feats,
                                           aggregator_type=args.agg_function, norm=None)
                            for _ in range(len(self.edges))
                        ])),
                        aggregate='stack'
                    )
                )
        else:
            for _ in range(args.conv_layers):
                self.conv_layers.append(
                    dglnn.HeteroGraphConv(
                        dict(zip(self.edges, [
                            dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats,
                                           aggregator_type=args.agg_function, norm=None)
                            for _ in range(len(self.edges))
                        ])),
                        aggregate='stack'
                    )
                )

        # Conv activations/norms
        self.conv_acts = nn.ModuleList()
        self.conv_norm = nn.ModuleList()
        if args.activation == 'gelu':
            self.conv_acts += [nn.GELU() for _ in range(args.conv_layers * 2)]
        elif args.activation == 'prelu':
            self.conv_acts += [nn.PReLU() for _ in range(args.conv_layers * 2)]
        elif args.activation == 'relu':
            self.conv_acts += [nn.ReLU() for _ in range(args.conv_layers * 2)]
        elif args.activation == 'leaky_relu':
            self.conv_acts += [nn.LeakyReLU() for _ in range(args.conv_layers * 2)]

        if args.normalization == 'batch':
            self.conv_norm += [nn.BatchNorm1d(hid_feats) for _ in range(args.conv_layers * len(self.edges))]
        elif args.normalization == 'layer':
            self.conv_norm += [nn.LayerNorm(hid_feats) for _ in range(args.conv_layers * len(self.edges))]
        elif args.normalization == 'group':
            self.conv_norm += [nn.GroupNorm(4, hid_feats) for _ in range(args.conv_layers * len(self.edges))]

        # Readout MLP(s)
        self.readout_linears = nn.ModuleList()
        self.readout_acts = nn.ModuleList()

        if args.weighted_sum:
            print("Weighted_sum enabled. Argument '--no_readout_concatenate' won't take effect.")
            for _ in range(args.readout_layers - 1):
                self.readout_linears.append(nn.Linear(hid_feats, hid_feats))
            self.readout_linears.append(nn.Linear(hid_feats, out_feats))
        elif self.nrc:
            for _ in range(args.readout_layers - 1):
                self.readout_linears.append(nn.Linear(hid_feats, hid_feats))
            self.readout_linears.append(nn.Linear(hid_feats, out_feats))
        else:
            for _ in range(args.readout_layers - 1):
                self.readout_linears.append(nn.Linear(hid_feats * args.conv_layers, hid_feats * args.conv_layers))
            self.readout_linears.append(nn.Linear(hid_feats * args.conv_layers, out_feats))

        if args.activation == 'gelu':
            self.readout_acts += [nn.GELU() for _ in range(args.readout_layers - 1)]
        elif args.activation == 'prelu':
            self.readout_acts += [nn.PReLU() for _ in range(args.readout_layers - 1)]
        elif args.activation == 'relu':
            self.readout_acts += [nn.ReLU() for _ in range(args.readout_layers - 1)]
        elif args.activation == 'leaky_relu':
            self.readout_acts += [nn.LeakyReLU() for _ in range(args.readout_layers - 1)]

        # Layer weights for weighted-sum readout
        self.wt = nn.Parameter(torch.zeros(args.conv_layers))
        
        # Post-readout linear calibration per-protein: y' = a*y + b
        self.calibrate = getattr(args, "calibrate", True)
        self.calib_a = nn.Parameter(torch.ones(out_feats))
        self.calib_b = nn.Parameter(torch.zeros(out_feats))

    # ----- one conv layer -----
    def conv(self, graph, layer, h, hist):
        args = self.args
        h0 = hist[-1]
        h = self.conv_layers[layer](graph, h, mod_kwargs=dict(
            zip(self.edges, [{
                'edge_weight': F.dropout(graph.edges[self.edges[i]].data['weight'],
                                         p=args.edge_dropout, training=self.training)
            } for i in range(len(self.edges))])
        ))

        h_feat = h['feature'].squeeze(1)
        h_cell = h['cell'].squeeze(1)

        h_feat_proc = self.conv_acts[layer * 2](self.conv_norm[layer * len(self.edges) + 1](h_feat))
        h_cell_proc = self.conv_acts[layer * 2 + 1](self.conv_norm[layer * len(self.edges)](h_cell))

        if args.model_dropout > 0:
            h = {
                'feature': F.dropout(h_feat_proc, p=args.model_dropout, training=self.training),
                'cell': F.dropout(h_cell_proc, p=args.model_dropout, training=self.training)
            }
        else:
            h = {
                'feature': h_feat_proc,
                'cell': h_cell_proc
            }
        return h

    # ----- initial embeddings -----
    def calculate_initial_embedding(self, graph):
        args = self.args

        feat_ids = graph.nodes['feature'].data['id'].long().view(-1) 
        cell_ids = graph.nodes['cell'].data['id'].long().view(-1)

        input1 = F.leaky_relu(self.embed_feat(feat_ids))
        input2 = F.leaky_relu(self.embed_cell(cell_ids))
        
        if not args.no_batch_features:
            batch_features = graph.nodes['cell'].data['bf'].to(input2.device).float()
            input2 += F.leaky_relu(F.dropout(self.extra_encoder(batch_features), p=0.2, training=self.training))

        hfeat = input1
        hcell = input2
        for i in range(args.embedding_layers - 1, (args.embedding_layers - 1) * 2):
            hfeat = self.input_linears[i](hfeat)
            hfeat = self.input_acts[i](hfeat)
            if args.normalization != 'none':
                hfeat = self.input_norm[i](hfeat)
            if args.model_dropout > 0:
                hfeat = F.dropout(hfeat, p=args.model_dropout, training=self.training)

        for i in range(args.embedding_layers - 1):
            hcell = self.input_linears[i](hcell)
            hcell = self.input_acts[i](hcell)
            if args.normalization != 'none':
                hcell = self.input_norm[i](hcell)
            if args.model_dropout > 0:
                hcell = F.dropout(hcell, p=args.model_dropout, training=self.training)

        return hfeat, hcell

    # ----- propagation (sampling version) -----
    def propagate_with_sampling(self, blocks):
        args = self.args
        hfeat, hcell = self.calculate_initial_embedding(blocks[0])
        h = {'feature': hfeat, 'cell': hcell}

        for i in range(args.conv_layers):
            if i > 0:
                hfeat0, hcell0 = self.calculate_initial_embedding(blocks[i])
                h = {'feature': torch.cat([h['feature'], hfeat0], 1),
                     'cell': torch.cat([h['cell'], hcell0], 1)}
            hist = [h]
            h = self.conv(blocks[i], i, h, hist)

        hist = [h] * (args.conv_layers + 1)
        return hist

    # ----- propagation (full graph) -----
    def propagate(self, graph):
        args = self.args
        hfeat, hcell = self.calculate_initial_embedding(graph)
        h = {'feature': hfeat, 'cell': hcell}
        hist = [h]

        for i in range(args.conv_layers):
            if i == 0 or args.residual == 'none':
                pass
            elif args.residual == 'res_add':
                if args.initial_residual:
                    h = {'feature': h['feature'] + hist[0]['feature'],
                         'cell': h['cell'] + hist[0]['cell']}
                else:
                    h = {'feature': h['feature'] + hist[-2]['feature'],
                         'cell': h['cell'] + hist[-2]['cell']}
            elif args.residual == 'res_cat':
                if args.initial_residual:
                    h = {'feature': torch.cat([h['feature'], hist[0]['feature']], 1),
                         'cell': torch.cat([h['cell'], hist[0]['cell']], 1)}
                else:
                    h = {'feature': torch.cat([h['feature'], hist[-2]['feature']], 1),
                         'cell': torch.cat([h['cell'], hist[-2]['cell']], 1)}

            h = self.conv(graph, i, h, hist)
            hist.append(h)

        return hist

    # ----- forward -----
    def forward(self, graph, sampled: bool = False, embedding: bool = False):
        args = self.args
        if sampled:
            hist = self.propagate_with_sampling(graph)
        else:
            hist = self.propagate(graph)

        if args.weighted_sum:
            h = 0
            weight = torch.softmax(self.wt, -1)
            for i in range(args.conv_layers):
                h += weight[i] * hist[i + 1]['cell']
        elif not self.nrc:
            h = torch.cat([i['cell'] for i in hist[1:]], 1)
        else:
            h = hist[-1]['cell']

        if embedding:
            return h

        for i in range(args.readout_layers - 1):
            h = self.readout_linears[i](h)
            h = F.dropout(self.readout_acts[i](h), p=args.model_dropout, training=self.training)
        h = self.readout_linears[-1](h)

        # Optional linear calibration per output protein
        if self.calibrate:
            h = h * self.calib_a.unsqueeze(0) + self.calib_b.unsqueeze(0)

        if args.output_relu == 'relu':
            return F.relu(h)
        elif args.output_relu == 'leaky_relu':
            return F.leaky_relu(h)
        return h