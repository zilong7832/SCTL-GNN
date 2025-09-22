import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from pathlib import Path

def to_dense(X):
    """Convert matrix to dense numpy array (support sparse input)."""
    if hasattr(X, "toarray"):
        return X.toarray()
    if hasattr(X, "todense"):
        return np.asarray(X.todense())
    return np.asarray(X)

def clr_transform(mat, axis=1, pseudocount=1.0):
    """Perform CLR (centered log-ratio) transformation."""
    X = np.array(mat, dtype=float) + pseudocount
    if axis == 1:  # per row (cell-wise)
        gm = np.exp(np.mean(np.log(X), axis=1, keepdims=True))
        return np.log(X / gm)
    elif axis == 0:  # per column (feature-wise)
        gm = np.exp(np.mean(np.log(X), axis=0, keepdims=True))
        return np.log(X / gm)
    else:
        raise ValueError("axis must be 0 or 1")

def extract_protein_matrix_and_names(adata: ad.AnnData):
    """
    Extract protein expression matrix and names.
    Priority: obsm['protein_expression'] + uns['protein_name'].
    Fallback: use adata.X + var_names.
    """
    if "protein_expression" in adata.obsm:
        M = to_dense(adata.obsm["protein_expression"])
        names = adata.uns.get("protein_name", None)
        if names is None:
            names = [f"ADT{i}" for i in range(M.shape[1])]
        else:
            names = list(map(str, names))
        return M, names
    return to_dense(adata.X), adata.var_names.astype(str).tolist()

# ========================
# Pipeline steps (unchanged)
# ========================

def cluster_rna(adata: ad.AnnData, n_pcs=50, leiden_res=0.5, cluster_key="groups"):
    """Cluster RNA data using PCA → neighbors → Leiden."""
    A = adata.copy()
    if "counts" in A.layers:
        A.X = A.layers["counts"].copy()
    sc.pp.pca(A, n_comps=min(n_pcs, A.n_vars - 1))
    sc.pp.neighbors(A)
    sc.tl.leiden(A, key_added=cluster_key, resolution=leiden_res)
    return A

def normalize_rna(adata: ad.AnnData, target_sum=1e4):
    """Normalize RNA counts using normalize_total + log1p."""
    A = adata.copy()
    if "counts" in A.layers:
        A.X = A.layers["counts"].copy()
    sc.pp.normalize_total(A, target_sum=target_sum)
    sc.pp.log1p(A)
    return A

def normalize_protein(adata: ad.AnnData, pseudocount=1.0):
    """Normalize protein data with CLR transform."""
    M, prot_names = extract_protein_matrix_and_names(adata)
    M_clr = clr_transform(M, axis=1, pseudocount=pseudocount)
    var = pd.DataFrame(index=pd.Index(prot_names, name="protein"))
    return ad.AnnData(X=M_clr, obs=adata.obs.copy(), var=var)

# ========================
# Main pipeline (MODIFIED)
# ========================

def run_pipeline():
    # -------- Parameters (edit here) --------
    input_h5ad  = Path("/mnt/scratch/zhan2210/datasets/different samples/CITE-PBMC-Li/Group2.h5ad")
    outdir      = Path("/mnt/scratch/zhan2210/datasets/different samples/CITE-PBMC-Li/")
    prefix      = "Group2" # A prefix for the output file names

    # --- Preprocessing settings ---
    n_pcs       = 50
    leiden_res  = 0.5
    target_sum  = 10000
    pseudocount = 1.0
    # ----------------------------------------

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {input_h5ad}")
    adata = ad.read_h5ad(input_h5ad)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy() 

    # Step 1: Cluster
    # print("Step 1: Clustering RNA...")
    # adata_clustered = cluster_rna(adata, n_pcs=n_pcs, leiden_res=leiden_res)

    # Step 2: Normalize RNA and Protein data
    print("Step 2: Normalizing RNA & Protein...")
    processed_rna = normalize_rna(adata, target_sum=target_sum)
    processed_protein = normalize_protein(adata, pseudocount=pseudocount)

    # Step 3: Export the two processed AnnData objects
    print(f"Step 3: Exporting results to: {outdir}")
    rna_out_path = outdir / f"{prefix}.processed_rna.h5ad"
    protein_out_path = outdir / f"{prefix}.processed_protein.h5ad"

    processed_rna.write_h5ad(rna_out_path, compression="gzip")
    processed_protein.write_h5ad(protein_out_path, compression="gzip")
    
    print(f"\nExported 2 files:")
    print(f"  RNA: {rna_out_path}")
    print(f"  Protein: {protein_out_path}")
    print("\nPipeline finished successfully. ✨")

if __name__ == "__main__":
    run_pipeline()