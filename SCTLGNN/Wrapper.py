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
            Fine-tunes the model on a specific cell type with sample-wise weighted MSE loss.
            This version saves the best model based on validation loss and reloads it after training.
            """
            device = self.args.device
            g = g.to(device)
            y = y.to(device)

            for name, param in self.model.named_parameters():
                if name.startswith("embed_feat") or name.startswith("embed_cell"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            opt = torch.optim.AdamW(trainable_params, lr=ft_lr, weight_decay=self.args.weight_decay)

            best_val_loss, epochs_no_improve = float('inf'), 0
            
            best_ft_dict = deepcopy(self.model.state_dict())

            for epoch in range(ft_epochs):
                self.model.train()
                
                out = self.model(g)
                
                train_indices = split["train"]
                losses = F.mse_loss(out[train_indices], y[train_indices], reduction="none")
                
                w_train = sample_weights[train_indices].to(device).unsqueeze(1)
                weighted_loss = (losses * w_train).mean()

                opt.zero_grad()
                weighted_loss.backward()
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                opt.step()
                
                torch.cuda.empty_cache()

                val_loss = self.score(g, split["valid"], y[split["valid"]], device)
                
                if verbose and epoch % eval_interval == 0:
                    log_msg = f"[Fine-tune: {celltype}] Epoch {epoch+1:03d}: Train Loss={weighted_loss.item():.4f}, Val Loss={val_loss:.4f}"
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
                        print(f"Early stopping at epoch {epoch+1} for cell type '{celltype}'. Best val loss: {best_val_loss:.4f}")
                        break
            
            print(f"Finished fine-tuning for {celltype}. Loading best model with val loss: {best_val_loss:.4f}")
            self.model.load_state_dict(best_ft_dict)
            
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