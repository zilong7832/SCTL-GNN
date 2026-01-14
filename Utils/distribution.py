# -*- coding: utf-8 -*-
import anndata
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import issparse

# --- 将你提供的 load_data 函数复制到这里 ---
def load_data(dataset_name: str):
    """Loads input and target AnnData objects based on the dataset name."""
    
    # ===== 根据数据集名称选择路径 =====
    if dataset_name == "SLN-111":
        base_dir = "/mnt/scratch/zhan2210/datasets/different samples/CITE-SLN111-Gayoso"
        train_prefix = "Mouse1"
        test_prefix = "Mouse2"

    elif dataset_name == "SLN-208":
        base_dir = "/mnt/scratch/zhan2210/datasets/different samples/CITE-SLN208-Gayoso"
        train_prefix = "Mouse1"
        test_prefix = "Mouse2"

    elif dataset_name == "PBMC-Li":
        base_dir = "/mnt/scratch/zhan2210/datasets/different samples/CITE-PBMC-Li"
        train_prefix = "Group1"
        test_prefix = "Group2"

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    print(f"Loading {dataset_name} train data...")
    # ===== 读取训练集 =====
    input_train = anndata.read_h5ad(f"{base_dir}/{train_prefix}.processed_rna.h5ad")
    target_train = anndata.read_h5ad(f"{base_dir}/{train_prefix}.processed_protein.h5ad")
    
    return input_train, target_train

def get_metrics(adata: anndata.AnnData):
    """
    计算 AnnData 对象的总表达量和平均表达量
    """
    # 1. 计算每个细胞的总表达量 (文库大小)
    if issparse(adata.X):
        # 对于稀疏矩阵
        total_counts_per_cell = np.asarray(adata.X.sum(axis=1)).flatten()
        mean_expr_per_feature = np.asarray(adata.X.mean(axis=0)).flatten()
    else:
        # 对于密集矩阵
        total_counts_per_cell = np.sum(adata.X, axis=1)
        mean_expr_per_feature = np.mean(adata.X, axis=0)
        
    # 过滤掉 0 (在log图中会出问题)
    total_counts_per_cell = total_counts_per_cell[total_counts_per_cell > 0]
    mean_expr_per_feature = mean_expr_per_feature[mean_expr_per_feature > 0]

    return total_counts_per_cell, mean_expr_per_feature

# *** 修改点 1: 增加一个 output_dir 参数 ***
def plot_dataset_distributions(dataset_name: str, output_dir: str):
    """
    加载指定数据集并绘制其 RNA 和 Protein 的分布图
    """
    try:
        rna_adata, protein_adata = load_data(dataset_name)
        
        print(f"Calculating metrics for {dataset_name}...")
        rna_counts, rna_means = get_metrics(rna_adata)
        prot_counts, prot_means = get_metrics(protein_adata)

        print(f"Plotting {dataset_name}...")
        
        # --- 创建 2x2 子图 ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Distributions for Dataset: {dataset_name} (Train Set, log scale)", fontsize=18, y=1.02)
        
        # --- 1. RNA 细胞总表达量 ---
        sns.histplot(rna_counts, ax=axes[0, 0], kde=True, log_scale=True, bins=100)
        axes[0, 0].set_title("RNA: Total Counts per Cell")
        axes[0, 0].set_xlabel("Log(Total Counts)")
        axes[0, 0].set_ylabel("Frequency")
        
        # --- 2. Protein 细胞总表达量 ---
        sns.histplot(prot_counts, ax=axes[0, 1], kde=True, log_scale=True, bins=100, color='orange')
        axes[0, 1].set_title("Protein: Total Counts per Cell")
        axes[0, 1].set_xlabel("Log(Total Counts)")
        axes[0, 1].set_ylabel("Frequency")

        # --- 3. RNA 特征平均表达量 ---
        sns.histplot(rna_means, ax=axes[1, 0], kde=True, log_scale=True, bins=100)
        axes[1, 0].set_title("RNA: Mean Expression per Feature")
        axes[1, 0].set_xlabel("Log(Mean Expression)")
        axes[1, 0].set_ylabel("Frequency")

        # --- 4. Protein 特征平均表达量 ---
        sns.histplot(prot_means, ax=axes[1, 1], kde=True, log_scale=True, bins=100, color='orange')
        axes[1, 1].set_title("Protein: Mean Expression per Feature")
        axes[1, 1].set_xlabel("Log(Mean Expression)")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()
        
        # *** 修改点 2: 使用 output_dir 来构建保存路径 ***
        # 1. 确保目标目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. 创建完整的文件路径
        output_filename = f"{dataset_name}_distributions.png"
        full_output_path = os.path.join(output_dir, output_filename)
        
        # 3. 保存图像到指定路径
        plt.savefig(full_output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {full_output_path}")
        # *** 修改结束 ***
        
        plt.close(fig) # 关闭图像释放内存

    except Exception as e:
        print(f"Could not process dataset {dataset_name}. Error: {e}")

def main():
    """
    主函数：循环遍历所有数据集并绘图
    """
    # 设置 Seaborn 风格
    sns.set_theme(style="whitegrid")
    
    # *** 修改点 3: 定义你的目标路径 ***
    target_output_directory = "/mnt/home/zhan2210/scProtein/code/plot"
    
    datasets_to_plot = ["SLN-111", "SLN-208", "PBMC-Li"]
    
    for ds_name in datasets_to_plot:
        print(f"\n--- Processing {ds_name} ---")
        # *** 修改点 4: 将路径传递给绘图函数 ***
        plot_dataset_distributions(ds_name, target_output_directory)
    
    print("\nAll plotting complete.")

if __name__ == "__main__":
    main()