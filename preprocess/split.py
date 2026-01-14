import anndata
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# --- 配置 ---
N_SEEDS = 10
N_SPLITS = 5
LABEL_COL = "celltypes"  # 对应 R 脚本中生成的 COL_FINAL

# 定义数据集配置
# 注意：这里需要填入 R 脚本输出的实际路径
DATASET_CONFIGS = [
    {
        # R脚本输出的 RNA Input 路径
        "path_rna": "/mnt/scratch/zhan2210/data/CITE-PBMC-Li/PBMC-Li_clean_input_rna.h5ad",
        # R脚本输出的 Protein Target 路径
        "path_prot": "/mnt/scratch/zhan2210/data/CITE-PBMC-Li/PBMC-Li_clean_target_protein_clr.h5ad",
        # 输出拆分文件的目录
        "output_dir": "/mnt/scratch/zhan2210/data/PBMC-Li_clean_split",
        "name": "PBMC-Li_Clean"
    },
    {
        # R脚本输出的 RNA Input 路径
        "path_rna": "/mnt/scratch/zhan2210/data/CITE-SLN111-Gayoso/SLN-111_clean_input_rna.h5ad",
        # R脚本输出的 Protein Target 路径
        "path_prot": "/mnt/scratch/zhan2210/data/CITE-SLN111-Gayoso/SLN-111_clean_target_protein_clr.h5ad",
        # 输出拆分文件的目录
        "output_dir": "/mnt/scratch/zhan2210/data/SLN111_clean_split",
        "name": "SLN111_Clean"
    },
        {
        # R脚本输出的 RNA Input 路径
        "path_rna": "/mnt/scratch/zhan2210/data/PBMC/PBMC_clean_input_rna.h5ad",
        # R脚本输出的 Protein Target 路径
        "path_prot": "/mnt/scratch/zhan2210/data/PBMC/PBMC_clean_target_protein_clr.h5ad",
        # 输出拆分文件的目录
        "output_dir": "/mnt/scratch/zhan2210/data/PBMC_clean_split",
        "name": "PBMC_Clean"
    },
            {
        # R脚本输出的 RNA Input 路径
        "path_rna": "/mnt/scratch/zhan2210/data/CITE-SLN208-Gayoso/SLN-208_clean_input_rna.h5ad",
        # R脚本输出的 Protein Target 路径
        "path_prot": "/mnt/scratch/zhan2210/data/CITE-SLN208-Gayoso/SLN-208_clean_target_protein_clr.h5ad",
        # 输出拆分文件的目录
        "output_dir": "/mnt/scratch/zhan2210/data/SLN208_clean_split",
        "name": "SLN208_Clean"
    },
]

for config in DATASET_CONFIGS:
    PATH_RNA = config["path_rna"]
    PATH_PROT = config["path_prot"]
    OUTPUT_DIR = config["output_dir"]
    DS_NAME = config["name"]
    
    print(f"\n=============================================")
    print(f"🔄 Starting split for dataset: {DS_NAME}")
    print(f"=============================================")

    # --- 1. 加载双模态数据 (Modified Load Part) ---
    if not os.path.exists(PATH_RNA) or not os.path.exists(PATH_PROT):
        print(f"!!! Error: Input files not found for {DS_NAME}.")
        print(f"    Missing: {PATH_RNA}")
        print(f"    OR:      {PATH_PROT}")
        continue
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading RNA data from: {PATH_RNA}")
    ad_rna = anndata.read_h5ad(PATH_RNA)
    
    print(f"Loading Protein data from: {PATH_PROT}")
    ad_prot = anndata.read_h5ad(PATH_PROT)

    # --- 校验对齐 ---
    # 确保 RNA 和 Protein 的细胞名称完全一致且顺序相同
    if not np.array_equal(ad_rna.obs_names, ad_prot.obs_names):
        print("⚠️ Warning: Index mismatch detected. Attempting to align intersection...")
        common_cells = ad_rna.obs_names.intersection(ad_prot.obs_names)
        ad_rna = ad_rna[common_cells].copy()
        ad_prot = ad_prot[common_cells].copy()
        print(f"   -> Aligned to {len(common_cells)} common cells.")
    
    if ad_rna.n_obs != ad_prot.n_obs:
        raise ValueError("Cell counts do not match even after alignment check!")

    print("Merging modalities for processing...")
    
    # --- 构建完备对象 ---
    # 将 Protein 数据放入 RNA 对象的 obsm 中，保持与后续代码兼容
    # R脚本导出的 Protein .X 已经是 CLR normalized data
    # 转换为 dense format (如果需要) 或者保持 sparse，视后续模型需求而定
    # 这里直接 copy，通常 anndata 会处理好
    
    # 核心修改：把 ad_prot.X 塞进 ad_rna.obsm['protein_expression']
    # 这样后续的 split 代码就能直接把 protein 数据带上
    
    # 如果 ad_prot.X 是 sparse matrix，某些 dataloader 可能需要 dense
    # 如果确定需要 dense 矩阵，可以使用: ad_prot.X.toarray() 
    ad_rna.obsm['protein_expression'] = ad_prot.X.copy()
    
    # 同时把 protein 的 feature names (抗体名) 保存下来，以防万一
    ad_rna.uns['protein_names'] = list(ad_prot.var_names)

    # 将合并后的对象命名为 data_full_merged，接管原有逻辑
    data_full_merged = ad_rna
    
    # 释放内存
    del ad_prot

    # --- 下面是原有的 Split 逻辑 (保持不变) ---
    
    if LABEL_COL not in data_full_merged.obs.columns:
        # 尝试处理一下常见的列名不匹配问题
        print(f"Available columns: {data_full_merged.obs.columns}")
        raise KeyError(f"Expected final label column '{LABEL_COL}' not found in AnnData.obs for {DS_NAME}.")

    # 提取分层标签和总细胞数
    labels = data_full_merged.obs[LABEL_COL].values
    n_obs = data_full_merged.n_obs

    print(f"Total cells: {n_obs}. Using '{LABEL_COL}' for stratified split.")
    print("Starting pre-splitting (ONLY FOLD=0)...")

    # --- 2. 循环处理10个 REPI ---
    for repi in range(N_SEEDS): # 循环 0, 1, ..., 9
        print(f"\n--- Processing REPI (Seed) = {repi} ---")
        
        # 初始化 KFold
        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=repi
        )
        
        # 只取 FOLD=0 (第一个拆分结果)
        split_generator = skf.split(np.arange(n_obs), labels)
        train_idx, test_idx = next(split_generator)
        fold = 0 
        
        print(f"  Fold 0 (Train: {len(train_idx)}, Test: {len(test_idx)})")

        # --- 3. 创建 data_train 和 data_test 对象 ---
        # 由于我们已经把 protein 放在了 obsm 中，这里切片会自动带上它
        data_train = data_full_merged[train_idx, :].copy()
        data_test  = data_full_merged[test_idx, :].copy()
        
        # 检查验证
        if 'protein_expression' not in data_train.obsm:
            print("[WARNING] 'protein_expression' missing in train subset!")
        
        # --- 4. 保存 FOLD=0 的文件 ---
        train_filename = f"train_repi{repi}_fold{fold}.h5ad"
        test_filename  = f"test_repi{repi}_fold{fold}.h5ad"
        
        train_path_out = os.path.join(OUTPUT_DIR, train_filename)
        test_path_out  = os.path.join(OUTPUT_DIR, test_filename)
        
        data_train.write_h5ad(train_path_out)
        data_test.write_h5ad(test_path_out)
        
        # 清理 generator
        del split_generator

    print(f"\n✓ Pre-splitting complete for {DS_NAME} (Only FOLD=0 for all REPIs).")
    print(f"All files saved to: {OUTPUT_DIR}")
    
print("\n\nAll datasets processed.")