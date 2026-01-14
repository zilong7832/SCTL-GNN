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
        """Loads input and target AnnData objects processed by the new pipeline."""
        import anndata
        from pathlib import Path

        # 初始化变量
        input_train, target_train = None, None
        input_test, target_test = None, None
        
        # 基础路径
        base_data_path = Path("/mnt/scratch/zhan2210/data")

        # ==========================================
        # 1. SLN-111 Dataset
        # ==========================================
        if dataset_name == "SLN-111":
            # 路径: .../data/CITE-SLN111-Gayoso/SLN-111_clean...
            data_dir = base_data_path / "CITE-SLN111-Gayoso"
            prefix = "SLN-111_clean" 
            batch_col = "batch_indices"

            input_train = anndata.read_h5ad(data_dir / f"{prefix}.processed_rna.h5ad")
            target_train = anndata.read_h5ad(data_dir / f"{prefix}.processed_protein.h5ad")

        # ==========================================
        # 2. SLN-208 Dataset
        # ==========================================
        elif dataset_name == "SLN-208":
            # 路径: .../data/CITE-SLN208-Gayoso/SLN-208_clean...
            data_dir = base_data_path / "CITE-SLN208-Gayoso"
            prefix = "SLN-208_clean"
            batch_col = "batch_indices"

            input_train = anndata.read_h5ad(data_dir / f"{prefix}.processed_rna.h5ad")
            target_train = anndata.read_h5ad(data_dir / f"{prefix}.processed_protein.h5ad")

        # ==========================================
        # 3. PBMC-Li Dataset
        # ==========================================
        elif dataset_name == "PBMC-Li":
            # 路径: .../data/CITE-PBMC-Li/PBMC-Li_clean...
            data_dir = base_data_path / "CITE-PBMC-Li"
            prefix = "PBMC-Li_clean"
            batch_col = "donor"  

            input_train = anndata.read_h5ad(data_dir / f"{prefix}.processed_rna.h5ad")
            target_train = anndata.read_h5ad(data_dir / f"{prefix}.processed_protein.h5ad")

        # ==========================================
        # 4. PBMC Dataset (Original)
        # ==========================================
        elif dataset_name == "PBMC":
            # 路径: .../data/PBMC/PBMC_clean...
            data_dir = base_data_path / "PBMC"
            prefix = "PBMC_clean"
            batch_col = "donor"

            input_train = anndata.read_h5ad(data_dir / f"{prefix}.processed_rna.h5ad")
            target_train = anndata.read_h5ad(data_dir / f"{prefix}.processed_protein.h5ad")

        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        # ==========================================
        # Common Post-Processing for Batch Info
        # ==========================================
        # 这里统一处理 batch 列，确保模型能读取
        for adata in [input_train, target_train]:
            if adata is not None:
                if batch_col in adata.obs.columns:
                    # 将指定的 batch 列 (如 batch_indices) 转为字符串并存入 "batch"
                    adata.obs["batch"] = adata.obs[batch_col].astype(str)
                else:
                    # 如果找不到 batch 列，抛出错误提示
                    raise KeyError(f"Column '{batch_col}' not found in obs for dataset {dataset_name}. "
                                    f"Ensure the preprocessing pipeline preserves '.obs'.")

        print(f"Successfully loaded {dataset_name}:")
        print(f"  Train cells: {input_train.shape[0]}")
        
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
            "celltypes" if "celltypes" in input_adata.obs.columns else None
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
        celltype_key: str = "celltypes", drop_zero_cells_in_train: bool = False,
        mode: Literal["correlation", "hvg"] = "correlation"  # <--- 新增参数
    ) -> Tuple[AnnData, AnnData]:
        """
        Select features based on either Target-Gene correlation OR Highly Variable Genes (HVG).
        
        Args:
            percent: 
                - If mode='correlation': Top percent of genes (e.g., 15 for 15%).
                - If mode='hvg': Number of top HVGs (e.g., 2000).
            mode: 'correlation' (supervised) or 'hvg' (unsupervised).
        """

        assert mod1_train.n_obs == mod2_train.n_obs, "Train sets must have the same number of cells."
        
        selected_genes = []

        #Branch 1: HVG Mode (Unsupervised)
        if mode == "hvg":
            import scanpy as sc
            print(f"[Feature Selection] Mode: HVG (Unsupervised). Selecting top {int(percent)} genes...")
            
            # 使用副本计算 HVG，避免污染原始数据（如 raw counts）
            adata_for_hvg = mod1_train.copy()
            
            # 简单的预处理以适应 Seurat flavor HVG 计算
            # 如果是 count 数据 (max > 50 是个粗略判断)，先归一化 + log1p
            if np.max(adata_for_hvg.X) > 50:
                sc.pp.normalize_total(adata_for_hvg, target_sum=1e4)
                sc.pp.log1p(adata_for_hvg)
            
            try:
                # 计算 HVG
                sc.pp.highly_variable_genes(
                    adata_for_hvg,
                    n_top_genes=int(percent), # 复用 percent 参数作为数量
                    flavor='seurat',
                    subset=False
                )
                # 提取基因名
                selected_genes = adata_for_hvg.var_names[adata_for_hvg.var['highly_variable']].tolist()
            except Exception as e:
                print(f"[WARN] HVG calculation failed: {e}. Fallback to all genes.")
                selected_genes = mod1_train.var_names.tolist()

        # Branch 2: Correlation Mode (Supervised, Original Logic)
        else:
            print(f"[Feature Selection] Mode: Correlation (Target-guided). Selecting top {percent}% genes...")
            assert celltype_key in mod1_train.obs.columns, f"'{celltype_key}' not found in mod1_train.obs"

            # Convert to dense
            X = mod1_train.X.toarray() if sp.issparse(mod1_train.X) else np.asarray(mod1_train.X)
            Y = mod2_train.X.toarray() if sp.issparse(mod2_train.X) else np.asarray(mod2_train.X)

            gene_counts = Counter()
            ct_series = mod1_train.obs[celltype_key]

            # Per-celltype correlation
            for ct in ct_series.unique():
                mask = (ct_series == ct).values
                if mask.sum() < 2:
                    continue

                X_ct, Y_ct = X[mask, :], Y[mask, :]

                if method == "pearson":
                    X_ct_centered = X_ct - X_ct.mean(axis=0)
                    Y_ct_centered = Y_ct - Y_ct.mean(axis=0)
                    cov = (X_ct_centered.T @ Y_ct_centered) / max(1, (X_ct.shape[0] - 1))
                    std_X = np.std(X_ct, axis=0, ddof=1)
                    std_Y = np.std(Y_ct, axis=0, ddof=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        corr = np.nan_to_num(cov / np.outer(std_X, std_Y), nan=0.0, posinf=0.0, neginf=0.0)

                elif method == "spearman":
                    Xr = np.apply_along_axis(rankdata, 0, X_ct)
                    Yr = np.apply_along_axis(rankdata, 0, Y_ct)
                    Xr_centered = Xr - Xr.mean(axis=0)
                    Yr_centered = Yr - Yr.mean(axis=0)
                    cov = (Xr_centered.T @ Yr_centered) / max(1, (Xr.shape[0] - 1))
                    std_X = np.std(Xr, axis=0, ddof=1)
                    std_Y = np.std(Yr, axis=0, ddof=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        corr = np.nan_to_num(cov / np.outer(std_X, std_Y), nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    raise ValueError("method must be 'pearson' or 'spearman'")

                mean_abs_corr = np.mean(np.abs(corr), axis=1)
                # 这里 percent 是百分比
                n_top = max(1, int(round(mod1_train.n_vars * (percent / 100.0))))
                top_idx = np.argpartition(-mean_abs_corr, kth=n_top - 1)[:n_top]
                gene_counts.update(mod1_train.var_names[top_idx])
            
            # Aggregate selected genes
            n_keep = max(1, int(round(mod1_train.n_vars * (percent / 100.0))))
            selected_genes = sorted(gene_counts, key=gene_counts.get, reverse=True)[:n_keep]

        # === Common Post-processing (Filtering & Safety Checks) ===
        
        # Intersection with Test set
        original_count = len(selected_genes)
        selected_genes = [g for g in selected_genes if g in mod1_train.var_names and g in mod1_test.var_names]
        
        if len(selected_genes) < original_count:
            print(f"   Filtered out {original_count - len(selected_genes)} genes not present in Test set.")

        if not selected_genes:
            print("[WARN] No genes selected! Defaulting to first gene to prevent crash.")
            selected_genes = [mod1_train.var_names[0]]

        # Apply filtering to train & test
        mod1_train_filtered = mod1_train[:, selected_genes].copy()
        mod1_test_filtered = mod1_test[:, selected_genes].copy()

        # Preserve obs metadata
        mod1_train_filtered.obs = mod1_train.obs.copy()
        mod1_test_filtered.obs = mod1_test.obs.copy()

        # Optionally drop zero-cells in train
        if drop_zero_cells_in_train:
            keep_rows_mask = np.array(mod1_train_filtered.X.sum(axis=1)).flatten() > 0
            n_dropped = mod1_train_filtered.n_obs - keep_rows_mask.sum()
            if n_dropped > 0:
                print(f"   Dropped {n_dropped} cells with zero expression in selected features.")
                mod1_train_filtered = mod1_train_filtered[keep_rows_mask, :].copy()
                # 注意：调用方需要手动同步 target_train (在 pipeline 里处理)

        return mod1_train_filtered, mod1_test_filtered
    
    @staticmethod
    def related_celltypes(
        adata: AnnData,
        celltype_key: str = "celltypes",
        top_n_markers: int = 50,
        top_k: int = 2,   # 新增：直接在这里决定 top-k
    ) -> Dict[str, Dict[str, float]]:
        """
        返回“已筛选”的相似度矩阵 gamma_matrix：
        - gamma_matrix[ct][ct] = 1.0
        - 对每个 ct 仅保留按 Jaccard 排序的前 top_k 个 related 的 γ（Jaccard 值）
        - 其它一律 γ=0
        - 若某个 ct 的样本数 >= 1000，则只保留自身 γ=1，其他全 0
        """
        # 计算各类的样本数
        if celltype_key not in adata.obs:
            raise KeyError(f"'{celltype_key}' not found in adata.obs")
        ct_counts = Counter(map(str, adata.obs[celltype_key].values))

        # 1) 先算“完整”的 Jaccard
        sc.tl.rank_genes_groups(
            adata, groupby=celltype_key, method="wilcoxon",
            n_genes=top_n_markers, pts=True, use_raw=False, layer=None
        )
        markers_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
        types = [str(c) for c in markers_df.columns]
        marker_sets = {ct: set(markers_df[ct]) for ct in types}

        full_jaccard = {ct: {} for ct in types}
        for i, ct1 in enumerate(types):
            for j, ct2 in enumerate(types):
                if i == j:
                    full_jaccard[ct1][ct2] = 1.0
                    continue
                inter = len(marker_sets[ct1] & marker_sets[ct2])
                union = len(marker_sets[ct1] | marker_sets[ct2])
                full_jaccard[ct1][ct2] = (inter / union) if union > 0 else 0.0

        # 2) 做“前置筛选”：对每个 ct 只保留 top-k，其余置 0；自身保留 1
        gamma_matrix = {ct: {k: 0.0 for k in types} for ct in types}
        for ct in types:
            n_ct = ct_counts.get(ct, 0)
            if n_ct >= 1000:
                # 大样本：只认自己
                gamma_matrix[ct][ct] = 1.0
                continue

            # 排序拿前 k（排除自己）
            pairs = [(other, float(full_jaccard[ct][other])) for other in types if other != ct]
            pairs.sort(key=lambda x: x[1], reverse=True)
            topk = [other for other, g in pairs if g > 0][:max(0, int(top_k))]
            # 写入：自己=1，top-k=各自 Jaccard，其它=0
            gamma_matrix[ct][ct] = 1.0
            for other in topk:
                gamma_matrix[ct][other] = float(full_jaccard[ct][other])

        return gamma_matrix
        
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
    def estimate_beta_from_data(mod1: AnnData, mod2: AnnData, celltype_key: str = "celltypes") -> float:
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
        similarity_matrix: Dict[str, Dict[str, float]],  # ← 已筛过：只有自身和 top-k 为正
        sample_counts: Dict[str, int],
        beta: float,
        input_dim: Optional[int],
        device: torch.device,
        gamma_threshold: float,  # 占位，已不使用
    ) -> torch.Tensor:
        """
        直接根据“已筛过”的 γ 矩阵计算权重：
        - 仅对 γ>0 的 related 参与；其它等同于排除
        - 仍保留 n_Q>=1000 的快速通道（全部权重给目标）
        """
        d = input_dim if input_dim is not None else 50
        n_Q = sample_counts.get(target_ct, 0)
        if n_Q >= 1000:
            weight_map = {ct: 0.0 for ct in sample_counts.keys()}
            weight_map[target_ct] = 1.0
            weights = [weight_map.get(ct, 0.0) for ct in obs_celltypes]
            return torch.tensor(weights, dtype=torch.float32, device=device).clamp_min(1e-12)

        # 目标半径与倒数
        r_Q = n_Q ** (-2.0 * beta / (2.0 * beta + d))
        inv_Q = (1.0 / r_Q) if r_Q > 0 else 0.0  # = n_Q^{2β/(2β+d)}

        # 仅取 γ>0 的 related
        inv_P = {}
        if target_ct in similarity_matrix:
            for other_ct, gamma in similarity_matrix[target_ct].items():
                if other_ct == target_ct:
                    continue
                g = float(gamma)
                if g <= 0.0:
                    continue
                n_P = sample_counts.get(other_ct, 0)
                if n_P <= 0:
                    continue
                exp = (2.0 * g * beta) / (2.0 * g * beta + d) if (2.0 * g * beta + d) > 0 else 0.0
                r_P = n_P ** (-exp)
                inv = (1.0 / r_P) if r_P > 0 else 0.0  # = n_P^{exp}
                if inv > 0:
                    inv_P[other_ct] = inv

        denom = inv_Q + sum(inv_P.values())
        weight_map = {ct: 0.0 for ct in sample_counts.keys()}
        if denom < 1e-12:
            weight_map[target_ct] = 1.0
        else:
            weight_map[target_ct] = inv_Q / denom
            for ct, inv in inv_P.items():
                weight_map[ct] = inv / denom

        weights = [weight_map.get(ct, 0.0) for ct in obs_celltypes]
        return torch.tensor(weights, dtype=torch.float32, device=device).clamp_min(1e-12)
        
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
            
    @staticmethod
    def get_finetune_indices_and_weights(
        input_train_full: AnnData,
        celltype_to_tune: str,
        similarity_matrix: Dict[str, Dict[str, float]],
        repi: int,
        args: argparse.Namespace,
        celltype_key: str = "celltypes"
    ) -> Tuple[Optional[dict], Optional[torch.Tensor], Optional[list], Optional[Counter]]:
        """
        获取微调所需的 *全局* 训练/验证索引和样本权重。
        - 不创建新图。
        - 索引是相对于全局图 g 的。
        - 权重 *只* 为训练索引 (ft_train_idx) 计算。
        """
        # 1) 确定 active cell types (逻辑不变)
        active_celltypes = []
        if celltype_to_tune in similarity_matrix:
            for other_ct, gamma in similarity_matrix[celltype_to_tune].items():
                if float(gamma) > 0.0:
                    active_celltypes.append(other_ct)
        if celltype_to_tune not in active_celltypes:
            active_celltypes = [celltype_to_tune] + active_celltypes
        print(f"Finding active cell indices for '{celltype_to_tune}', actives: {active_celltypes}")

        # 2) 筛选出 *全局训练索引*
        train_obs_full = input_train_full.obs
        active_train_mask_bool = train_obs_full[celltype_key].isin(active_celltypes).values
        active_global_train_indices = np.where(active_train_mask_bool)[0]

        if active_global_train_indices.size < 10: # 样本太少，无法划分
            print(f"Skipping '{celltype_to_tune}': Too few active training samples ({active_global_train_indices.size})")
            return None, None, None, None

        # 3) 从 "active" 索引中划分 train/validation (仍为全局索引)
        active_labels = train_obs_full.loc[active_train_mask_bool, celltype_key].values
        try:
            ft_train_idx, ft_valid_idx = train_test_split(
                active_global_train_indices,
                test_size=0.15, # 使用 15% 的 active 数据做验证
                stratify=active_labels,
                random_state=repi
            )
        except ValueError: # 样本太少无法分层
            ft_train_idx, ft_valid_idx = train_test_split(
                active_global_train_indices, test_size=0.15, random_state=repi
            )
        
        split_sub_global = {'train': ft_train_idx, 'valid': ft_valid_idx}
        celltype_counts_sub = Counter(active_labels) # 权重计算需要 *所有* active 细胞的计数

        # 4) *只* 为 ft_train_idx 计算样本权重
        obs_celltypes_ft_train = train_obs_full.iloc[ft_train_idx][celltype_key].tolist()
        
        sample_weights = Utils.loss_weights(
            obs_celltypes=obs_celltypes_ft_train, # <- 只传入训练集的细胞类型
            target_ct=celltype_to_tune,
            similarity_matrix=similarity_matrix,
            sample_counts=celltype_counts_sub, # <- 使用 active 细胞的总数
            beta=args.ft_beta,
            input_dim=args.hidden_size,
            device=args.device,
            gamma_threshold=args.ft_gamma_threshold,
        )
        # sample_weights 现在的长度 == len(ft_train_idx)

        return split_sub_global, sample_weights, obs_celltypes_ft_train, celltype_counts_sub
    
    @staticmethod
    def get_finetune_indices_and_weights_new(
        input_train_full: AnnData,
        celltype_to_tune: str,
        similarity_matrix: Dict[str, Dict[str, float]],
        repi: int,
        args: argparse.Namespace,
        celltype_key: str = "celltypes"
    ) -> Tuple[Optional[dict], Optional[torch.Tensor], Optional[list], Optional[Counter]]:
        """
        获取微调所需的 *全局* 训练/验证索引和样本权重。
        - [V3 逻辑] 验证集 (valid) 只包含目标细胞。
        - [V3 逻辑] 训练集 (train) 包含 (剩余的目标细胞 + 所有的相关细胞)。
        - 索引是相对于全局图 g 的。
        - 权重 *只* 为训练索引 (ft_train_idx) 计算。
        """
        train_obs_full = input_train_full.obs

        # 1) 确定 active cell types
        active_celltypes = []
        if celltype_to_tune in similarity_matrix:
            for other_ct, gamma in similarity_matrix[celltype_to_tune].items():
                if float(gamma) > 0.0:
                    active_celltypes.append(other_ct)
        if celltype_to_tune not in active_celltypes:
            active_celltypes = [celltype_to_tune] + active_celltypes
        
        # 2) 分别筛选 目标 (Target) 和 相关 (Related) 的全局训练索引
        target_mask = (train_obs_full[celltype_key] == celltype_to_tune).values
        target_global_indices = np.where(target_mask)[0]

        related_celltypes = [ct for ct in active_celltypes if ct != celltype_to_tune]
        related_mask = train_obs_full[celltype_key].isin(related_celltypes).values
        related_global_indices = np.where(related_mask)[0]

        # 3) 检查 Target 样本量是否足够划分
        # 至少需要2个 target 细胞才能划分出 train 和 valid
        if target_global_indices.size < 2: 
            print(f"Skipping '{celltype_to_tune}': Too few target training samples ({target_global_indices.size}) to create a validation set.")
            return None, None, None, None

        # 4) *只* 对 Target 索引进行 train/validation 划分
        try:
            # 确保验证集最少有 1 个样本
            val_split_ratio = getattr(args, "val_split_ratio", 0.15)
            n_valid = max(1, int(target_global_indices.size * val_split_ratio))
            
            # 确保训练集也至少有 1 个样本
            if target_global_indices.size - n_valid < 1:
                n_valid = target_global_indices.size - 1 # 给训练集留1个

            if n_valid <= 0: # 还是不够分 (比如总共只有1个)
                 raise ValueError("Not enough target samples to split.")

            ft_train_target_idx, ft_valid_target_idx = train_test_split(
                target_global_indices,
                test_size=n_valid, # 使用 15% 的 *target* 数据做验证
                random_state=repi
                # 注意：如果 target 样本少，stratify 可能会失败，故移除
            )
        except ValueError as e:
            # 降级：如果样本极少（例如只有2个），随机分
            print(f"Warning: Could not split target cells for {celltype_to_tune} ({e}). Splitting 1-and-1.")
            if target_global_indices.size < 2: return None, None, None, None # 再次确认
            
            # 手动分配，确保 train 和 valid 都有
            indices_shuffled = np.random.RandomState(repi).permutation(target_global_indices)
            split_point = 1
            ft_valid_target_idx = indices_shuffled[:split_point]
            ft_train_target_idx = indices_shuffled[split_point:]

        if ft_train_target_idx.size == 0 or ft_valid_target_idx.size == 0:
             print(f"Skipping '{celltype_to_tune}': Failed to create valid train/valid split from target cells ({target_global_indices.size}).")
             return None, None, None, None

        # 5) 组合最终的 split
        # 训练集 = (Target的训练部分) + (所有的Related)
        ft_train_idx = np.concatenate([ft_train_target_idx, related_global_indices])
        # 验证集 = (Target的验证部分)
        ft_valid_idx = ft_valid_target_idx

        split_sub_global = {'train': ft_train_idx, 'valid': ft_valid_idx}
        
        print(f"Split for '{celltype_to_tune}':")
        print(f"  Target Cells (Total): {target_global_indices.size}")
        print(f"  Related Cells (Total): {related_global_indices.size}")
        print(f"  FT Train (Target): {ft_train_target_idx.size}")
        print(f"  FT Valid (Target-Only): {ft_valid_target_idx.size}")
        print(f"  FT Train (Total Mix): {ft_train_idx.size}")

        # 6) 准备权重计算所需的数据
        # 权重计算需要 *所有* active 细胞的总计数
        active_mask = train_obs_full[celltype_key].isin(active_celltypes).values
        active_labels = train_obs_full.loc[active_mask, celltype_key].values
        celltype_counts_sub = Counter(active_labels) 
        
        # 权重计算需要 *ft_train_idx* (混合体) 对应的细胞类型列表
        obs_celltypes_ft_train = train_obs_full.iloc[ft_train_idx][celltype_key].tolist()
        
        # 7) *只* 为 ft_train_idx (混合体) 计算样本权重
        sample_weights = Utils.loss_weights(
            obs_celltypes=obs_celltypes_ft_train, # <- 传入 *新* 训练集的细胞类型
            target_ct=celltype_to_tune,
            similarity_matrix=similarity_matrix,
            sample_counts=celltype_counts_sub, # <- 仍使用 *所有* active 细胞的总数
            beta=args.ft_beta,
            input_dim=args.hidden_size,
            device=args.device,
            gamma_threshold=args.ft_gamma_threshold,
        )
        # sample_weights 现在的长度 == len(ft_train_idx)

        return split_sub_global, sample_weights, obs_celltypes_ft_train, celltype_counts_sub

    @staticmethod
    def get_finetune_test_indices_and_data(
        input_test_full: AnnData,
        target_test_full: AnnData,
        celltype_to_tune: str,
        train_size: int,
        celltype_key: str = "celltypes"
    ) -> Tuple[Optional[np.ndarray], Optional[AnnData]]:
        """
        获取微调所需的 *全局* 测试索引和匹配的 target_test anndata。
        - 不创建新图。
        - 索引是相对于全局图 g 的 (即带有 train_size 偏移)。
        """
        # 1) 仅筛选目标类型的 *测试* 细胞
        test_obs_full = input_test_full.obs
        target_test_mask_bool = (test_obs_full[celltype_key] == celltype_to_tune).values
        
        # 2) 获取本地索引 (相对于 input_test_full)
        local_test_indices = np.where(target_test_mask_bool)[0]
        
        if local_test_indices.size == 0:
            print(f"No test cells found for '{celltype_to_tune}'.")
            return None, None
        
        # 3) 加上偏移量，得到 *全局索引*
        global_test_indices = local_test_indices + train_size
        
        # 4) 获取匹配的 target anndata (用于评估和保存)
        target_test_sub = target_test_full[target_test_mask_bool].copy()
        
        return global_test_indices, target_test_sub
    
    @staticmethod
    def save_matrix_as_h5ad(
        X: np.ndarray,
        obs: pd.DataFrame,
        var_names: list,
        file_path: str
    ):
        X = np.asarray(X, dtype=np.float32)
        var_df = pd.DataFrame(index=pd.Index(var_names, name=None))
        adata = AnnData(X=X, obs=obs.copy(), var=var_df)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        adata.write_h5ad(file_path)
        print(f"[H5AD] Saved: {file_path} | shape={X.shape}")