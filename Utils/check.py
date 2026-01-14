import anndata

ad = anndata.read_h5ad("/mnt/scratch/zhan2210/data/CITE-SLN208-Gayoso/train_cite_SLN-208_input_rna.h5ad")
print(ad)
print(ad.shape)
print(ad.var.columns[:30])
print(ad.var.head())

# 最关键：看是否有 feature 类型标记
for c in ["feature_types", "feature_type", "modality", "gene_ids", "protein_ids"]:
    if c in ad.var.columns:
        print(c, ad.var[c].unique()[:20])