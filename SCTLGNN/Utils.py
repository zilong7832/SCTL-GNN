import os
import random
from collections import Counter
from typing import Tuple, Literal, Iterable, Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

import anndata
from anndata import AnnData
import scanpy as sc
from pathlib import Path

import torch
import dgl
import mudata
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from Data import Data
from BatchFeature import BatchFeature
from Graph import SCTLGNNGraph

class Utils:
    """A collection of utility functions for the sctlgnn pipeline."""

    @staticmethod
    def set_seed(seed: int, cuda: bool = True, extreme_mode: bool = False):
        """Set seeds for reproducibility across python, numpy, torch, and dgl."""
        # Set seeds for base python, numpy, and torch
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Set seeds for DGL
        dgl.seed(seed)
        dgl.random.seed(seed)

        # Set seeds for CUDA devices if available and enabled
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Configure cuDNN for deterministic behavior if in extreme_mode
        if extreme_mode:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def load_data(dataset_name: str):
        """Loads input and target AnnData objects based on the dataset name."""
        input_train = anndata.read_h5ad("/mnt/scratch/zhan2210/datasets/different samples/CITE-PBMC-Li/Group1.processed_rna.h5ad")
        target_train = anndata.read_h5ad("/mnt/scratch/zhan2210/datasets/different samples/CITE-PBMC-Li/Group1.processed_protein.h5ad")
        input_train.obs["batch"] = input_train.obs["donor"] 
        target_train.obs["batch"] = target_train.obs["donor"]
        
        input_test = anndata.read_h5ad("/mnt/scratch/zhan2210/datasets/different samples/CITE-PBMC-Li/Group2.processed_rna.h5ad")
        target_test = anndata.read_h5ad("/mnt/scratch/zhan2210/datasets/different samples/CITE-PBMC-Li/Group2.processed_protein.h5ad")
        input_test.obs["batch"] = input_test.obs["donor"] 
        target_test.obs["batch"] = target_test.obs["donor"]
        
        return input_train, target_train, input_test, target_test

    @staticmethod
    def split_data(
        input_adata,
        target_adata,
        # external test data, if provided will bypass split
        input_test=None,
        target_test=None,
        celltype_key: str = "celltype",
        n_splits: int = 5,
        fold: int = 0,
        repi: int = 1,
    ):
        """
        Split data into train and test sets.
        - If input_test and target_test are provided (not None), they will be used directly,
          and the input_adata/target_adata will be returned as train.
        - Otherwise, perform stratified k-fold split on input_adata/target_adata.
        Returns: (input_train, input_test, target_train, target_test)
        """
        # Case 1: external test data provided
        if input_test is not None and target_test is not None:
            input_train, target_train = input_adata, target_adata
            return input_train, input_test, target_train, target_test

        # Case 2: do k-fold split
        ct_key = celltype_key if celltype_key in input_adata.obs.columns else (
            "cell_type" if "cell_type" in input_adata.obs.columns else None
        )
        if ct_key is None:
            raise ValueError("No cell type column found (celltype or cell_type).")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repi)
        labels = input_adata.obs[ct_key].values
        train_idx, test_idx = next(
            (train, test) for i, (train, test) in enumerate(skf.split(np.arange(input_adata.n_obs), labels))
            if i == fold
        )
        input_train = input_adata[train_idx, :].copy()
        input_test  = input_adata[test_idx, :].copy()
        target_train = target_adata[train_idx, :].copy()
        target_test  = target_adata[test_idx, :].copy()

        return input_train, input_test, target_train, target_test

    @staticmethod
    def top_correlated_genes(
        mod1_train: AnnData, mod2_train: AnnData, mod1_test: AnnData,
        percent: float, method: Literal["pearson", "spearman"] = "pearson",
        celltype_key: str = "celltype", drop_zero_cells_in_train: bool = False
    ) -> Tuple[AnnData, AnnData]:
        """
        Selects top correlated genes from the training set and applies the filter to both train and test sets.
        """
        assert mod1_train.n_obs == mod2_train.n_obs, "Train sets must have the same number of cells."
        
        X = mod1_train.X.toarray() if sp.issparse(mod1_train.X) else np.asarray(mod1_train.X)
        Y = mod2_train.X.toarray() if sp.issparse(mod2_train.X) else np.asarray(mod2_train.X)

        gene_counts = Counter()
        ct_series = mod1_train.obs[celltype_key]

        for ct in ct_series.unique():
            mask = (ct_series == ct).values
            if mask.sum() < 2: continue

            X_ct, Y_ct = X[mask, :], Y[mask, :]
            
            if method == "pearson":
                X_ct_centered = X_ct - X_ct.mean(axis=0)
                Y_ct_centered = Y_ct - Y_ct.mean(axis=0)
                corr_matrix = (X_ct_centered.T @ Y_ct_centered) / (X_ct.shape[0] - 1)
            elif method == "spearman":
                X_ct_ranked = np.apply_along_axis(rankdata, 0, X_ct)
                Y_ct_ranked = np.apply_along_axis(rankdata, 0, Y_ct)
                X_ct_centered = X_ct_ranked - X_ct_ranked.mean(axis=0)
                Y_ct_centered = Y_ct_ranked - Y_ct_ranked.mean(axis=0)
                corr_matrix = (X_ct_centered.T @ Y_ct_centered) / (X_ct_ranked.shape[0] - 1)
            else:
                raise ValueError("Method must be 'pearson' or 'spearman'")

            std_X = np.std(X_ct, axis=0, ddof=1)
            std_Y = np.std(Y_ct, axis=0, ddof=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = np.nan_to_num(corr_matrix / np.outer(std_X, std_Y))
            
            mean_abs_corr = np.mean(np.abs(corr), axis=1)
            n_top = max(1, int(round(mod1_train.n_vars * (percent / 100.0))))
            top_idx = np.argpartition(-mean_abs_corr, kth=n_top - 1)[:n_top]
            gene_counts.update(mod1_train.var_names[top_idx])

        n_keep = max(1, int(round(mod1_train.n_vars * (percent / 100.0))))
        selected_genes = sorted(gene_counts, key=gene_counts.get, reverse=True)[:n_keep]
        selected_genes = [g for g in selected_genes if g in mod1_train.var_names and g in mod1_test.var_names]
        if not selected_genes: selected_genes = [mod1_train.var_names[0]]

        mod1_train_filtered = mod1_train[:, selected_genes].copy()
        mod1_test_filtered = mod1_test[:, selected_genes].copy()
        
        mod1_train_filtered.obs = mod1_train.obs.copy()
        mod1_test_filtered.obs = mod1_test.obs.copy()

        if drop_zero_cells_in_train:
            keep_rows_mask = np.array(mod1_train_filtered.X.sum(axis=1)).flatten() > 0
            mod1_train_filtered = mod1_train_filtered[keep_rows_mask, :].copy()
            # Note: Caller must sync target data if this option is used.
        
        return mod1_train_filtered, mod1_test_filtered
    
    @staticmethod
    def related_celltypes(adata: AnnData, celltype_key: str = "cell_type", top_n_markers: int = 50):
        """
        Calculates a pairwise Jaccard similarity matrix between all cell types based on their marker genes.
        """
        sc.tl.rank_genes_groups(adata, groupby=celltype_key, method="wilcoxon", n_genes=top_n_markers, pts=True)
        markers_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
        
        types = markers_df.columns
        marker_sets = {ct: set(markers_df[ct]) for ct in types}
        similarity_matrix = {ct: {} for ct in types}
        for i in range(len(types)):
            for j in range(i, len(types)):
                ct1 = types[i]
                ct2 = types[j]
                
                if i == j:
                    similarity_matrix[ct1][ct2] = 1.0
                else:
                    intersection_size = len(marker_sets[ct1] & marker_sets[ct2])
                    union_size = len(marker_sets[ct1] | marker_sets[ct2])
                    jaccard = intersection_size / union_size if union_size > 0 else 0
                    similarity_matrix[ct1][ct2] = jaccard
                    similarity_matrix[ct2][ct1] = jaccard
                
        return similarity_matrix
        
    @staticmethod
    def prepare_graph_and_data(
        mod1: anndata.AnnData,
        mod2: anndata.AnnData,
        train_size: int,
        args: argparse.Namespace
    ):
        # 1. Create MuData and set configuration
        mdata = mudata.MuData({"mod1": mod1, "mod2": mod2})
        data = Data(mdata, train_size=train_size)    
        data.set_config(
            feature_mod="mod1",
            label_mod="mod2",
            feature_channel_type="X",
            label_channel_type="X"
        )

        # 2. Build the graph 
        data = SCTLGNNGraph(
            inductive=args.inductive,
            cell_init=args.cell_init,
        )(data)

        # 3. Optionally add batch features
        if not args.no_batch_features:
            BatchFeature()(data.data.mod['mod1'])

        # 4. Create a train/validation split
        val_split_ratio = getattr(args, "val_split_ratio", 0.15)
        idx = np.random.permutation(train_size)
        split_idx = max(1, int(len(idx) * val_split_ratio))  # 保证至少1个valid
        split = {"train": idx[:-split_idx], "valid": idx[-split_idx:]}

        # 5. Collect dataset info
        data_info = {
            "FEATURE_SIZE": mod1.shape[1],
            "TRAIN_SIZE": train_size,
            "OUTPUT_SIZE": mod2.shape[1],
            "CELL_SIZE": mod1.shape[0],
            "BATCH_NUM": 0
        }

        # 6. Extract graphs
        if args.inductive:
            g, gtest = data.data.uns["g"], data.data.uns["gtest"]
        else:
            g = gtest = data.data.uns["g"]

        # 7. Extract labels
        _, y_train = data.get_train_data(return_type="torch")
        _, y_test = data.get_test_data(return_type="torch")

        # 8. Add batch features if enabled
        if not args.no_batch_features and "batch_features" in data.data["mod1"].obsm:
            batch_features = torch.as_tensor(
                data.data["mod1"].obsm["batch_features"], dtype=torch.float32
            )
            data_info["BATCH_NUM"] = batch_features.shape[1]
            if args.inductive:
                g.nodes["cell"].data["bf"] = batch_features[:train_size]
                gtest.nodes["cell"].data["bf"] = batch_features
            else:
                g.nodes["cell"].data["bf"] = batch_features

        return g, gtest, y_train, y_test, split, data_info
    
    @staticmethod
    def estimate_beta_from_data(mod1: AnnData, mod2: AnnData, celltype_key: str = "celltype.l1") -> float:
        """
        Estimates the beta parameter by modeling the relationship between
        distances in the input space (mod1) and the target space (mod2).
        """
        n_sample = min(1000, mod1.shape[0])
        try:
            sample_idx, _ = train_test_split(
                np.arange(mod1.shape[0]),
                train_size=n_sample,
                stratify=mod1.obs[celltype_key]
            )
        except ValueError:
            sample_idx = np.random.choice(np.arange(mod1.shape[0]), n_sample, replace=False)

        mod1_sample = mod1[sample_idx, :].copy()
        mod2_sample = mod2[sample_idx, :].copy()
        X_sample = mod1_sample.X.toarray() if hasattr(mod1_sample.X, "toarray") else mod1_sample.X
        Y_sample = mod2_sample.X.toarray() if hasattr(mod2_sample.X, "toarray") else mod2_sample.X
        dists_X = euclidean_distances(X_sample)
        dists_Y = euclidean_distances(Y_sample)
        mask = ~np.eye(n_sample, dtype=bool)
        log_dx = np.log(dists_X[mask] + 1e-6)
        log_dy = np.log(dists_Y[mask] + 1e-6)
        reg = LinearRegression().fit(log_dx.reshape(-1, 1), log_dy)
        estimated_beta = float(reg.coef_[0])
        return max(0.01, min(estimated_beta, 1.0))
    @staticmethod
    def loss_weights(
        obs_celltypes: Iterable[str],
        target_ct: str,
        similarity_matrix: Dict[str, Dict[str, float]],
        sample_counts: Dict[str, int],
        beta: float,
        input_dim: Optional[int],
        device: torch.device,
        gamma_threshold: float,
    ) -> torch.Tensor:
        """
        Compute sample-wise weights for transfer learning.

        This version dynamically determines the number of related cell types (k)
        using a two-step process:
        1. First, it filters for related cell types with a similarity (gamma)
           score above `gamma_threshold`.
        2. Then, among these candidates, it uses the "elbow method" to find the
           optimal number of cell types to use by identifying the largest
           drop-off point in similarity scores.
        """
        n_Q = sample_counts.get(target_ct, 0)
        if n_Q >= 1000:
            weight_map = {ct: 0.0 for ct in sample_counts.keys()}
            weight_map[target_ct] = 1.0
            weights = [weight_map.get(ct, 0.0) for ct in obs_celltypes]
            w = torch.tensor(weights, dtype=torch.float32, device=device)
            return w.clamp_min(1e-12)
        candidates_after_threshold = []
        if target_ct in similarity_matrix:
            for other_ct, gamma in similarity_matrix[target_ct].items():
                if other_ct != target_ct and gamma > gamma_threshold:
                    candidates_after_threshold.append((gamma, other_ct))
        candidates_after_threshold.sort(key=lambda x: x[0], reverse=True)
        selected_related_cts = []
        num_candidates = len(candidates_after_threshold)

        if num_candidates == 0:
            pass
        elif num_candidates <= 2:
            selected_related_cts = [ct for _, ct in candidates_after_threshold]
        else:
            scores = [gamma for gamma, _ in candidates_after_threshold]
            score_diffs = np.diff(scores)
            elbow_index = np.argmax(score_diffs)
            dynamic_k = elbow_index + 1
            top_candidates = candidates_after_threshold[:dynamic_k]
            selected_related_cts = [ct for _, ct in top_candidates]

        d = input_dim if input_dim is not None else 50
        
        r_Q = n_Q ** (-2 * beta / (2 * beta + d))

        r_P: Dict[str, float] = {}
        for ct in selected_related_cts:
            n_P = sample_counts.get(ct, 0)
            if n_P == 0: continue
            
            gamma = similarity_matrix[target_ct][ct]
            exponent = -2 * gamma * beta / (2 * gamma * beta + d)
            r_P[ct] = n_P ** exponent
        
        inv_r_Q = 1.0 / r_Q if r_Q > 0 else 0.0
        inv_r_P = {ct: 1.0 / r for ct, r in r_P.items() if r > 0}

        denom = inv_r_Q + sum(inv_r_P.values())
        
        if denom < 1e-9:
            weight_map = {ct: 0.0 for ct in sample_counts.keys()}
            weight_map[target_ct] = 1.0
        else:
            weight_map = {ct: 0.0 for ct in sample_counts.keys()}
            weight_map[target_ct] = inv_r_Q / denom
            for ct, inv_r in inv_r_P.items():
                weight_map[ct] = inv_r / denom
            
        weights = [weight_map.get(ct, 0.0) for ct in obs_celltypes]
        w = torch.tensor(weights, dtype=torch.float32, device=device).clamp_min(1e-12)
        
        return w
    
    @staticmethod
    def save_evaluation_results(
        results: dict,
        output_folder: str,
        prefix: str
    ):
        """
        Save evaluation results into two consolidated CSVs per run (append mode).
        - <run_prefix>_protein_metrics.csv : columns [protein, pcc, rmse, celltype]
        - <run_prefix>_cell_metrics.csv    : columns [cell, pcc, rmse, celltype]

        The run_prefix (e.g., DATASET_REPI_FOLD) and celltype are both inferred
        from the input `prefix`. All fine-tuning results for different celltypes
        within the same run will be appended to the same pair of files.
        """
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # --- Step 1: Parse the prefix to get the filename base and the celltype ---
        tokens = str(prefix).split("_")

        # The "base prefix" for the filename is the first 3 parts: DATASET_REPI_FOLD
        if len(tokens) >= 3:
            prefix_base = "_".join(tokens[:3])
        else:
            # Fallback if prefix is malformed (e.g., during pre-training)
            prefix_base = prefix

        # The celltype is everything after the 3rd part
        if len(tokens) > 3:
            # Reconstruct celltype name, replacing underscores from the prefix with spaces
            celltype = " ".join(tokens[3:])
        else:
            celltype = "ALL"

        # --- Step 2: Construct the dynamic filenames based on the base prefix ---
        protein_csv_path = os.path.join(output_folder, f"{prefix_base}_protein_metrics.csv")
        cell_csv_path    = os.path.join(output_folder, f"{prefix_base}_cell_metrics.csv")

        # Helper function to safely extract a numeric pandas Series
        def _get_metric_series(df_or_series):
            if isinstance(df_or_series, pd.Series):
                return df_or_series
            if isinstance(df_or_series, pd.DataFrame):
                if df_or_series.shape[1] == 1:
                    return df_or_series.iloc[:, 0]
                for col in df_or_series.columns:
                    if pd.api.types.is_numeric_dtype(df_or_series[col]):
                        return df_or_series[col]
            return None

        # --- Step 3: Process and save protein metrics ---
        protein_pcc_series = _get_metric_series(results.get("protein_pcc"))
        protein_rmse_series = _get_metric_series(results.get("protein_rmse"))

        if protein_pcc_series is not None or protein_rmse_series is not None:
            idx = protein_pcc_series.index if protein_pcc_series is not None else protein_rmse_series.index
            df_protein = pd.DataFrame(index=idx)
            if protein_pcc_series is not None: df_protein["pcc"] = protein_pcc_series
            if protein_rmse_series is not None: df_protein["rmse"] = protein_rmse_series
            df_protein.reset_index(inplace=True)
            df_protein.rename(columns={"index": "protein"}, inplace=True)
            df_protein["celltype"] = celltype

            header_protein = not Path(protein_csv_path).exists()
            df_protein.to_csv(protein_csv_path, index=False, mode="a", header=header_protein)

        # --- Step 4: Process and save cell metrics ---
        cell_pcc_series = _get_metric_series(results.get("cell_pcc"))
        cell_rmse_series = _get_metric_series(results.get("cell_rmse"))

        if cell_pcc_series is not None or cell_rmse_series is not None:
            idx = cell_pcc_series.index if cell_pcc_series is not None else cell_rmse_series.index
            df_cell = pd.DataFrame(index=idx)
            if cell_pcc_series is not None: df_cell["pcc"] = cell_pcc_series
            if cell_rmse_series is not None: df_cell["rmse"] = cell_rmse_series
            df_cell.reset_index(inplace=True)
            df_cell.rename(columns={"index": "cell"}, inplace=True)
            df_cell["celltype"] = celltype

            header_cell = not Path(cell_csv_path).exists()
            df_cell.to_csv(cell_csv_path, index=False, mode="a", header=header_cell)