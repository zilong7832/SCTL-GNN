# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from pathlib import Path

# ========================
# Utils
# ========================

def to_dense(X):
    """Convert matrix to dense numpy array (support sparse input)."""
    if hasattr(X, "toarray"):
        return X.toarray()
    if hasattr(X, "todense"):
        return np.asarray(X.todense())
    return np.asarray(X)

# ========================
# RNA Processing (Normalization)
# ========================

def normalize_rna(adata: ad.AnnData) -> ad.AnnData:
    """
    处理 RNA：
      - 输入应为 Raw Counts (来自 R 的 input_rna.h5ad)
      - 若 layers["counts"] 不存在，将 X 备份进去
      - 执行：Library Size Normalization (至中位数) + log1p
      - 输出：
          obsm["rna_raw"]   = 原始计数
          layers["lognorm"] = 归一化后数据
          X                 = 归一化后数据
    """
    A = adata.copy()

    # 1. 确保有 counts 层 (如果 R 输出只有 X，这里备份一下)
    if "counts" not in A.layers:
        A.layers["counts"] = A.X.copy()
    
    # 2. 保存原始数据到 obsm (模型通常需要 raw 用于计算 loss)
    A.obsm["rna_raw"] = to_dense(A.layers["counts"])

    # 3. 计算 Library Size 中位数
    if hasattr(A.X, "A1"): # sparse
        lib = A.X.sum(axis=1).A1
    else:
        lib = np.asarray(A.X.sum(axis=1)).ravel()
    
    target_sum = float(np.median(lib))
    print(f"  -> RNA Normalization target sum (median): {target_sum:.2f}")

    # 4. 归一化 + Log1p
    sc.pp.normalize_total(A, target_sum=target_sum)
    sc.pp.log1p(A)

    # 5. 保存处理后的层
    A.layers["lognorm"] = A.X.copy()
    
    return A

# ========================
# Protein Processing (Formatting)
# ========================

def format_protein(adata: ad.AnnData, source_is_clr=False) -> ad.AnnData:
    """
    格式化蛋白数据：
      - R 代码输出的通常已经是 CLR 归一化后的数据 (.X)
      - 此函数将其整理为模型需要的格式
      - 输出：
          X: 保持输入值 (如果是 R 的输出，就是 CLR 值)
          layers["counts"]: 这里的命名是为了兼容模型读取，实际上存的是 R 处理后的值
          var: index 设为 protein name
    """
    # R 输出的数据在 .X 中，且转置处理过了 (Cells x Proteins)
    M = to_dense(adata.X)
    
    # 获取蛋白名称
    names = adata.var_names.astype(str).tolist()
    
    # 构建新的 AnnData 确保干净
    A = ad.AnnData(
        X=M,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(names, name="protein"))
    )

    # 模型通常寻找 'raw_counts' 或 'counts' 层
    # 注意：因为 R 已经做了 CLR，这里存的其实是 CLR 值，但为了代码兼容性保持结构
    A.layers["raw_counts"] = A.X.copy()
    
    # 也可以存一份到 raw
    A.raw = A.copy()
    
    if source_is_clr:
        print("  -> Note: Input protein data seems to be CLR normalized (from R). Structure preserved.")
        
    return A

# ========================
# Main pipeline (Connecting to R output)
# ========================

def run_pipeline():
    # -------- Parameters (Match R Output) --------
    # R 输出目录
    data_dir = Path("/mnt/scratch/zhan2210/data/PBMC")
    prefix   = "PBMC_clean" # R script 中的 PREFIX_OUTPUT
    
    # R 生成的具体文件名
    rna_input_path  = data_dir / f"{prefix}_input_rna.h5ad"
    prot_input_path = data_dir / f"{prefix}_target_protein_clr.h5ad"
    
    # 最终输出路径
    outdir = data_dir
    # ---------------------------------------------

    if not rna_input_path.exists():
        raise FileNotFoundError(f"R output not found: {rna_input_path}\nPlease run the R script first.")

    print(f"Loading RNA from R output: {rna_input_path}")
    adata_rna = ad.read_h5ad(rna_input_path)
    
    print(f"Loading Protein from R output: {prot_input_path}")
    adata_prot = ad.read_h5ad(prot_input_path)

    # 简单的对齐检查
    if adata_rna.shape[0] != adata_prot.shape[0]:
        raise ValueError("Error: Cell counts in RNA and Protein files do not match!")

    # ===== Step 1: 处理 RNA =====
    # R 输出的是 Raw Counts，所以这里我们需要进行 Normalize + Log1p
    print("\nStep 1: Normalizing RNA (Median + Log1p)...")
    processed_rna = normalize_rna(adata_rna)

    # ===== Step 2: 格式化 Protein =====
    # R 输出的是 CLR Normalized Data，我们将其格式化以便 Python 模型读取
    print("\nStep 2: Formatting Protein (Using CLR from R)...")
    processed_protein = format_protein(adata_prot, source_is_clr=True)

    # ===== Step 3: 导出最终文件 =====
    rna_out_path = outdir / f"{prefix}.processed_rna.h5ad"
    protein_out_path = outdir / f"{prefix}.processed_protein.h5ad"

    print("\nStep 3: Saving processed files...")
    processed_rna.write_h5ad(rna_out_path, compression="gzip")
    processed_protein.write_h5ad(protein_out_path, compression="gzip")

    print("\n✅ Pipeline finished.")
    print(f"  Processed RNA saved to:    {rna_out_path}")
    print(f"  Processed Protein saved to:{protein_out_path}")
    print("\nSummary:")
    print(f"  Cells: {processed_rna.shape[0]}")
    print(f"  Genes: {processed_rna.shape[1]}")
    print(f"  Proteins: {processed_protein.shape[1]}")

if __name__ == "__main__":
    run_pipeline()