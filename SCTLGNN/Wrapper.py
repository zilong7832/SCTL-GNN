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

import csv, time
from pathlib import Path


# =============================================================================
# [新增] GASDU 管理器 (V2 内存优化版)
# =============================================================================
class GasduManager:
    """
    Manages the GASDU (Gauss-Southwell Dynamic Update) logic.
    
    [V2 - Memory Optimized Selection]
    This class handles the periodic refresh and reuse of the gradient mask
    as described in Algorithm 1 of the paper.
    
    It implements a streaming *selection* to find the Top-k
    threshold, avoiding the torch.cat() of all gradients.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        k_percent: float,
        m_period: int,
        device: torch.device
    ):
        """
        Initializes the GASDU manager.

        Args:
            model: The model being fine-tuned (must have .args attribute for grad_clip).
            optimizer: The optimizer (e.g., AdamW) used for training.
            k_percent: The percentage of parameters to update (e.g., 0.01).
            m_period: The refresh period (in epochs/steps) for the mask.
            device: The torch device (e.g., 'cuda').
        """
        self.model = model
        self.optimizer = optimizer
        self.k_percent = k_percent
        self.m_period = m_period
        self.device = device
        
        # This dictionary will store the boolean mask for each parameter
        # e.g., { 'param_name': torch.Tensor([True, False, ...]) }
        self.mask: Dict[str, torch.Tensor] = {}
        
        # Get total number of trainable parameters
        self.total_trainable_params = self._get_total_trainable_params()
        if self.total_trainable_params == 0:
            raise ValueError("[GASDU] Model has no trainable parameters.")
            
        # k (k_count) is the absolute number of parameters to update
        self.k_count = max(1, int(self.total_trainable_params * (self.k_percent / 100.0)))
        
        print(f"[GasduManager V2] Initialized. k={self.k_count} ({self.k_percent}%) | "
              f"M_period={self.m_period} | "
              f"Total Trainable Params={self.total_trainable_params}")

    def _get_total_trainable_params(self) -> int:
        """Helper to count trainable parameters."""
        count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                count += param.numel()
        return count

    @torch.no_grad()
    def _refresh_mask(self):
        """
        [V2] Refreshes the gradient mask using a memory-efficient streaming *selection*.
        
        This finds the global Top-k threshold by iterating through parameter
        gradients one by one, using only an O(k) tensor pool, thus avoiding
        the torch.cat() of all gradients.
        
        This corresponds to Algorithm 1, lines 4-5.
        """
        print(f"[GASDU V2] Refreshing mask (k={self.k_count}) using streaming selection...")
        
        # 1. Initialize the O(k) candidate pool for Top-k values.
        # We find the k-th largest by sorting and picking the smallest.
        # 使用 .float() 确保类型正确
        top_k_pool = torch.full((self.k_count,), -1.0, device=self.device, dtype=torch.float32)
        current_threshold = -1.0
        
        # --- Pass 1: Find the global Top-k threshold ---
        for name, param in self.model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
                
            # Get flat, absolute gradients for this parameter
            g_flat = param.grad.abs().detach().flatten()
            
            # Find contenders from this param that are larger than our
            # current k-th largest value (the minimum of the pool).
            contenders = g_flat[g_flat > current_threshold]
            
            if contenders.numel() > 0:
                # Combine our current pool with the new contenders
                # 确保 contenders 也是 float32
                combined = torch.cat([top_k_pool, contenders.float()])
                
                # Sort and keep only the new k-largest
                # This is the most expensive part of the loop
                top_k_pool = torch.topk(combined, self.k_count, sorted=False).values
                
                # The new threshold is the smallest value in our k-largest pool
                current_threshold = top_k_pool.min()
                
            del g_flat, contenders

        # After Pass 1, `current_threshold` holds the true k-th largest gradient magnitude
        final_threshold = current_threshold.item()
        del top_k_pool # We only needed this to find the threshold
        
        if final_threshold < 0:
            print("[GASDU] Warning: Gradient threshold is negative. All gradients might be zero.")
            final_threshold = 0.0

        print(f"[GASDU V2] Pass 1 complete. Global gradient threshold={final_threshold:.2e}")

        # --- Pass 2: Create the boolean masks ---
        self.mask.clear() # Clear the old mask
        total_masked_params = 0
        for name, param in self.model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            
            # Create boolean mask based on the global threshold
            mask = (param.grad.abs().detach() >= final_threshold)
            self.mask[name] = mask.to(self.device) # Store the mask
            total_masked_params += mask.sum().item()

        print(f"[GASDU V2] Pass 2 complete. Mask created. "
              f"Actual params in mask={total_masked_params} (Target k={self.k_count})")


    @torch.no_grad()
    def _apply_mask_to_grads(self):
        """
        Applies the stored mask to the model's current gradients in-place.
        This corresponds to Algorithm 1, line 9.
        """
        if not self.mask:
            print("[GASDU] Warning: Tried to apply an empty mask. Skipping.")
            return

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name in self.mask:
                    # Apply mask (element-wise multiplication)
                    param.grad.mul_(self.mask[name])
                else:
                    # This param was not present during last refresh (e.g., frozen)
                    # or had no gradient. Zero it out just in case.
                    param.grad.zero_()

    def step(self, loss: torch.Tensor, epoch: int):
        """
        Performs a full GASDU step: backward, mask refresh/reuse, and optimizer step.
        This function orchestrates the logic from Algorithm 1.
        
        Args:
            loss: The loss tensor for the current step.
            epoch: The current epoch number (or step number).
        """
        
        # 1. Calculate full gradients (Algorithm 1, line 4 or 9)
        self.optimizer.zero_grad()
        loss.backward() # This is the unavoidable gradient materialization step
        
        is_refresh_step = (epoch % self.m_period == 0)
        
        # 2. Refresh or Reuse Mask
        if is_refresh_step:
            # Algorithm 1, lines 4-5: Calculate and store new \Lambda_t
            self._refresh_mask()
        else:
            # Algorithm 1, line 7: Reuse \Lambda_{t-1} (i.e., do nothing)
            pass
            
        # 3. Apply Mask (Algorithm 1, line 9: \tilde{G}_t <- \Lambda_t \odot \nabla f(W_t))
        if self.mask:
            self._apply_mask_to_grads()
        else:
            if is_refresh_step:
                print("[GASDU] Warning: Mask is empty after refresh. Performing full gradient step.")
            # If mask is empty on a reuse step, something is wrong, but we proceed
            # with a full gradient step to avoid crashing.
            
        # 4. Optimizer Step (Algorithm 1, line 10)
        # We clip gradients *after* masking
        # 假设 model 实例上有 .args 属性
        grad_clip = getattr(self.model.args, "grad_clip", 0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
        self.optimizer.step()
# =============================================================================
# [新增] GASDU 管理器结束
# =============================================================================


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

    def _save_curve(self, rows, header, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

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
            preds = self.predict(g, idx, device)  # already includes the chosen output activation
            loss = F.mse_loss(preds, labels.to(device)).item()
        return math.sqrt(loss)

    # ---------- training ----------
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
        [MODIFIED] Disables calibration parameters during pre-training.
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

        # === [MODIFICATION] ===
        # 1. Force-disable calibration during forward pass for pre-training
        print("[Pre-train] Disabling calibration (if any) for pre-training.")
        self.model.calibrate = False 
        
        # 2. Filter calibration params out of the optimizer and freeze them
        pretrain_params = []
        for name, param in self.model.named_parameters():
            if name in ['calib_a', 'calib_b']:
                param.requires_grad = False
            else:
                param.requires_grad = True # Make sure others are trainable
                pretrain_params.append(param)
        
        opt = torch.optim.AdamW(pretrain_params, # <-- Use filtered list
                                lr=self.args.learning_rate,
                                weight_decay=self.args.weight_decay)
        # === [End MODIFICATION] ===
        
        criterion = nn.MSELoss()

        tr, val, te = [], [], []
        curve_rows = [] 
        
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
                # Note: We are optimizing pretrain_params
                torch.nn.utils.clip_grad_norm_(pretrain_params, grad_clip)

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
                
                curr_lr = opt.param_groups[0]['lr']
                te_rmse = te[-1] if (eval and (y_test is not None) and len(te) > 0) else float('nan')
                curve_rows.append([epoch, tr[-1], val[-1], te_rmse, curr_lr, int(time.time())])

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
        
        if dataset is None: dataset = "NA"
        if repi is None: repi = -1
        if fold is None: fold = -1
        curve_path = f"{self.args.log_folder}/{self.args.prefix}_{dataset}_run{repi}_{fold}_fit_curve.csv"
        self._save_curve(
            curve_rows,
            header=["epoch","train_rmse","val_rmse","test_rmse","lr","ts"],
            out_path=curve_path
        )
        
        # === [MODIFICATION] ===
        # Restore grad and calibrate state so fine-tuning can use them
        print("[Pre-train] Re-enabling calibration param grads for fine-tuning.")
        for name, param in self.model.named_parameters():
            if name in ['calib_a', 'calib_b']:
                param.requires_grad = True
        self.model.calibrate = self.args.calibrate # Restore to user's setting
        # === [End MODIFICATION] ===
        
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
            eval_interval: int = 1,
            dataset: str = "NA",
            repi: int = -1,
            fold: int = -1,
            # [新增] 接收来自 pipeline 的 GASDU 参数
            k_percent: float = 100.0,
            m_period: int = 50
        ):
            """
            Fine-tunes the model using a pre-computed weight tensor for weighted MSE loss.
            [MODIFIED] Integrates GasduManager for optimization.
            [MODIFIED] Resets and enables calibration parameters at the start.
            """
            device = self.args.device
            g = g.to(device)
            y = y.to(device)
            
            # === [MODIFICATION] ===
            # Handle calibration logic at the START of fine-tuning for this celltype
            if self.args.calibrate:
                print(f"[Fine-tune: {celltype}] Enabling calibration and resetting parameters (a=1, b=0).")
                self.model.calibrate = True
                # Call the new reset method on the model
                self.model.reset_calibration_parameters() 
            else:
                self.model.calibrate = False
                print(f"[Fine-tune: {celltype}] Calibration is disabled.")
            # === [End MODIFICATION] ===
            
            if self.args.ft_freeze_embeddings:
                print("Freezing embedding layers for fine-tuning.") 
                for name, param in self.model.named_parameters():
                    # Freeze embeddings
                    if name.startswith("embed_feat") or name.startswith("embed_cell"):
                        param.requires_grad = False
                    else:
                        # Ensure all other params (including calib_a, calib_b) are trainable
                        param.requires_grad = True
                trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            else:
                print("Training all layers (including embeddings) during fine-tuning.")
                for _, param in self.model.named_parameters():
                    param.requires_grad = True
                trainable_params = self.model.parameters()
                
            # [修改] 将 trainable_params 转换为 list，以便 AdamW 正常接收
            # This list() conversion correctly includes calib_a and calib_b
            opt = torch.optim.AdamW(list(trainable_params), lr=ft_lr, weight_decay=self.args.weight_decay)
                
            best_val_loss, epochs_no_improve = float('inf'), 0
            best_ft_dict = deepcopy(self.model.state_dict())
            
            curve_rows = []
            
            train_idx = split["train"]
            
            # === [新增] Initialize GasduManager ===
            # 只有当 k_percent < 100% 时才启用 GASDU
            use_gasdu = (k_percent < 100.0 and k_percent > 0.0)

            gasdu_manager = None
            if use_gasdu:
                try:
                    gasdu_manager = GasduManager(
                        model=self.model,
                        optimizer=opt,
                        k_percent=k_percent,
                        m_period=m_period,
                        device=device
                    )
                except ValueError as e:
                    print(f"[GASDU] Error initializing manager: {e}. Falling back to full fine-tune.")
                    use_gasdu = False
            else:
                print("[GASDU] k_percent=100. Running standard full fine-tuning.")
            # === [新增] End ===
            
            # --- Fine-tuning Loop ---
            for epoch in range(ft_epochs):
                self.model.train()
                out = self.model(g)

                # Loss calculation (unchanged)
                losses_per_elem = F.mse_loss(out[train_idx], y[train_idx], reduction="none")
                w_train = sample_weights.to(device).unsqueeze(1)
                loss = (losses_per_elem * w_train).mean()

                # === [修改] Use GasduManager or standard optim step ===
                if gasdu_manager:
                    # GasduManager handles:
                    # 1. optimizer.zero_grad()
                    # 2. loss.backward()
                    # 3. Mask Refresh (if epoch % M == 0)
                    # 4. Mask Apply
                    # 5. grad_clip (via self.model.args)
                    # 6. optimizer.step()
                    gasdu_manager.step(loss, epoch)
                else:
                    # Original full fine-tune logic
                    opt.zero_grad()
                    loss.backward()
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    opt.step()
                # === [修改] End ===

                torch.cuda.empty_cache()

                # --- Validation and Early Stopping ---
                if epoch % eval_interval == 0:
                    val_loss = self.score(g, split["valid"], y[split["valid"]], device)

                    if verbose:
                        # Simplified logging message
                        log_msg = (
                            f"[Fine-tune: {celltype}] Epoch {epoch+1:03d}: "
                            f"Train Loss (MSE_w)={loss.item():.4f} | "
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
                            
                    curr_lr = opt.param_groups[0]['lr']
                    curve_rows.append([epoch, float(loss.item()), float(val_loss), curr_lr, int(time.time())])
            
            print(f"Finished fine-tuning for {celltype}. Loading best model with val loss: {best_val_loss:.4f}")
            self.model.load_state_dict(best_ft_dict)
            
            # [修复] 确保 safe_celltype 用于日志文件名
            safe_celltype_name = str(celltype).replace("/", "_").replace(" ", "")
            curve_path = f"{self.args.log_folder}/{self.args.prefix}_{dataset}_run{repi}_{fold}_{safe_celltype_name}_finetune_curve.csv"
            self._save_curve(
                curve_rows,
                header=["epoch","train_weighted_mse","val_rmse","lr","ts"],
                out_path=curve_path
            )
                
            # Unfreeze all parameters after fine-tuning for this cell type is complete
            # This is important in case ft_freeze_embeddings was True
            for param in self.model.parameters():
                param.requires_grad = True
                
            return self

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
            self.embed_cell = nn.Embedding(2, hid_feats)
        else:
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
        self.calibrate = getattr(self.args, "calibrate", False)
        self.calib_a = nn.Parameter(torch.ones(out_feats))
        self.calib_b = nn.Parameter(torch.zeros(out_feats))

    @torch.no_grad()
    def reset_calibration_parameters(self):
        """Resets calibration parameters to their initial state (a=1, b=0)."""
        print("[ScTlGNN] Resetting calibration parameters (a=1, b=0).")
        if hasattr(self, 'calib_a') and self.calib_a is not None:
            self.calib_a.fill_(1.0)
        if hasattr(self, 'calib_b') and self.calib_b is not None:
            self.calib_b.fill_(0.0) 

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

        input1 = F.leaky_relu(self.embed_feat(graph.srcdata['id']['feature']))
        input2 = F.leaky_relu(self.embed_cell(graph.srcdata['id']['cell'].long()))

        if not args.no_batch_features:
            batch_features = graph.srcdata['bf']['cell']
            input2 += F.leaky_relu(F.dropout(self.extra_encoder(batch_features), p=0.2,
                                             training=self.training))[:input2.shape[0]]

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

        # Unified output activation (training & evaluation share this)
        act_choice = getattr(self.args, "output_relu", "none")
        if act_choice == "relu":
            h = F.relu(h)
        elif act_choice == "leaky_relu":
            h = F.leaky_relu(h)
        elif act_choice == "softplus":
            h = F.softplus(h)
        # else: "none" — no activation

        # Optional per-protein linear calibration (default OFF)
        if getattr(self, "calibrate", False):
            h = h * self.calib_a.unsqueeze(0) + self.calib_b.unsqueeze(0)

        return h