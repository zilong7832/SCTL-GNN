import anndata
import os
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

# --- 配置 ---
# 输出目录
OUTPUT_MERGED_DIR = "/mnt/scratch/zhan2210/data"

# 要处理的数据集列表
DATASET_NAMES = ["SLN-111", "SLN-208", "PBMC-Li"]

# --- 数据加载与合并函数 ---
def load_and_merge_data(dataset_name: str) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    加载数据集的所有部分（例如 Mouse1, Mouse2），将它们进行纵向合并。
    注意：原始文件是“混合对象”：RNA 在 X/var，蛋白在 obsm['protein_expression']，蛋白名在 uns['protein_name']。
    返回两个独立的对象：
    1. input_full: 合并后的完整 RNA 数据 (AnnData: X=RNA)
    2. target_full: 合并后的完整 Protein 数据 (AnnData: X=Protein)
    """

    # 路径和列名的配置
    CONFIG = {
        "SLN-111": {
            "base": "/mnt/scratch/zhan2210/data/CITE-SLN111-Gayoso",
            "parts": [("Mouse1", "Mouse1"), ("Mouse2", "Mouse2")],
            "batch_col": "batch_indices",
        },
        "SLN-208": {
            "base": "/mnt/scratch/zhan2210/data/CITE-SLN208-Gayoso",
            "parts": [("Mouse1", "Mouse1"), ("Mouse2", "Mouse2")],
            "batch_col": "batch_indices",
        },
        "PBMC-Li": {
            "base": "/mnt/scratch/zhan2210/data/CITE-PBMC-Li",
            "parts": [("Group1", "Group1"), ("Group2", "Group2")],
            "batch_col": "donor",
        },
    }

    if dataset_name not in CONFIG:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    conf = CONFIG[dataset_name]

    input_list = []
    prot_mat_list = []
    obs_list = []

    protein_names_ref = None

    print(f"Loading and merging parts for {dataset_name}...")

    for rna_prefix, protein_prefix in conf["parts"]:
        # 读取混合文件（RNA在X，Protein在obsm）
        ad = anndata.read_h5ad(f"{conf['base']}/{rna_prefix}.h5ad")
        print(f"  Loaded {rna_prefix} (RNA shape: {ad.shape})")

        # --- basic checks ---
        if "protein_expression" not in ad.obsm:
            raise ValueError(f"{rna_prefix}.h5ad missing obsm['protein_expression'] (can't extract protein).")

        if "protein_name" not in ad.uns:
            raise ValueError(f"{rna_prefix}.h5ad missing uns['protein_name'] (can't name proteins).")

        # 统一 protein_name 成 list[str]
        prot_names = ad.uns["protein_name"]
        if isinstance(prot_names, (np.ndarray,)):
            prot_names = prot_names.tolist()
        # 有的可能是 pandas Index
        if hasattr(prot_names, "tolist"):
            prot_names = prot_names.tolist()

        if protein_names_ref is None:
            protein_names_ref = prot_names
        else:
            if list(prot_names) != list(protein_names_ref):
                raise ValueError(f"Protein panel mismatch between parts. {rna_prefix} has different protein_name list.")

        # 1) RNA：直接拿整个 AnnData（X=RNA）
        input_list.append(ad)

        # 2) Protein：先把矩阵取出来，后面手动 vstack（避免 concat 把 obsm 搞乱）
        prot_mat = ad.obsm["protein_expression"]
        prot_mat_list.append(prot_mat)

        # 3) obs：为构建 target_full 用（保证跟 input_full 细胞顺序一致）
        obs_list.append(ad.obs.copy())

    # 1. 合并 RNA（cells 纵向拼接）
    # join="outer" 保证基因并集；如果你确定基因完全一致，也可以改成 "inner"
    input_full = anndata.concat(input_list, join="outer")

    # 2. 合并 Protein matrix（cells 纵向拼接）
    # 兼容 numpy / sparse
    try:
        import scipy.sparse as sp
        if any(sp.issparse(m) for m in prot_mat_list):
            prot_mat_list = [m if sp.issparse(m) else sp.csr_matrix(m) for m in prot_mat_list]
            prot_full_X = sp.vstack(prot_mat_list, format="csr")
        else:
            prot_full_X = np.vstack(prot_mat_list)
    except Exception:
        # 没有 scipy 的情况下（一般你环境有），退化成 numpy
        prot_full_X = np.vstack([np.asarray(m) for m in prot_mat_list])

    # 3. 合并 obs（顺序与 prot_full_X 对齐）
    target_obs = pd.concat(obs_list, axis=0)
    target_obs.index = input_full.obs_names  # 强制对齐到 input_full 的细胞顺序/ID

    # 4. 构建 protein 的 var
    target_var = pd.DataFrame(index=pd.Index(protein_names_ref, name="protein"))

    # 5. 构建 target_full（X=Protein）
    target_full = anndata.AnnData(X=prot_full_X, obs=target_obs, var=target_var)

    # 6. 确保批次信息存在 (Batch info)
    batch_col = conf["batch_col"]
    for adata in [input_full, target_full]:
        if batch_col in adata.obs.columns:
            adata.obs["batch"] = adata.obs[batch_col].astype(str)
        else:
            print(f"Warning: Column '{batch_col}' not found in obs. (No batch column set)")

    print(f"Merge complete. Full RNA shape: {input_full.shape}, Full Protein shape: {target_full.shape}")

    # 7. 同步 metadata：把 target_full 里有而 input_full 没有的列补过去
    for col in target_full.obs.columns:
        if col not in input_full.obs.columns:
            input_full.obs[col] = target_full.obs[col]
            if col == "celltypes":
                print("  -> Synced 'celltypes' from Protein to RNA object.")

    return input_full, target_full


# --- 主执行流程 ---
os.makedirs(OUTPUT_MERGED_DIR, exist_ok=True)

for ds_name in DATASET_NAMES:
    print(f"\n=============================================")
    print(f"🚀 Starting merging for dataset: {ds_name}")
    print(f"=============================================")

    # A. 加载和合并
    try:
        input_full, target_full = load_and_merge_data(ds_name)
    except Exception as e:
        print(f"!!! Error loading {ds_name}: {e}")
        continue

    # B. 保存为两个独立的文件 (Input 和 Target)

    # 1. 保存 Input (RNA)
    input_filename = f"train_cite_{ds_name}_input_rna.h5ad"
    input_path = os.path.join(OUTPUT_MERGED_DIR, input_filename)
    print(f"Saving INPUT (RNA) file to: {input_path}")
    input_full.write_h5ad(input_path)

    # 2. 保存 Target (Protein)
    target_filename = f"train_cite_{ds_name}_target_protein.h5ad"
    target_path = os.path.join(OUTPUT_MERGED_DIR, target_filename)
    print(f"Saving TARGET (Protein) file to: {target_path}")
    target_full.write_h5ad(target_path)

    print(f"✓ Saved {ds_name} as split Input/Target files.")

print("\n\nAll requested datasets processed.")