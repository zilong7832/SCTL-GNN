import os, re
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.stats import pearsonr, gaussian_kde
from itertools import islice
import scanpy as sc

# =====================
# Paths & Models (请确保路径正确)
# =====================
MODELS = [
    ("Model-1 (Seurat)",
     "/mnt/scratch/zhan2210/output/Seurat/prediction/PBMC-Li_1_0_groundtruth.h5ad",
     "/mnt/scratch/zhan2210/output/Seurat/prediction/PBMC-Li_1_0_prediction.h5ad"),
    ("Model-2 (scmognn)",
     "/mnt/scratch/zhan2210/output/scmognn/dance_PBMC-Li_1_0_PBMC-Li_1_0/results/PBMC-Li_1_0_groundtruth.h5ad",
     "/mnt/scratch/zhan2210/output/scmognn/dance_PBMC-Li_1_0_PBMC-Li_1_0/results/PBMC-Li_1_0_prediction.h5ad"),
    ("Model-3 (sctlgnn)",
     "/mnt/scratch/zhan2210/output/sctlgnn/prediction/PBMC-Li_1_0_ALL_groundtruth.h5ad",
     "/mnt/scratch/zhan2210/output/sctlgnn/prediction/PBMC-Li_1_0_ALL_prediction.h5ad"),

]

# 输出目录
OUT_DIR  = "/mnt/home/zhan2210/scProtein/code/plot/detailed-analysis-plots-top5/PBMC-Li"
os.makedirs(OUT_DIR, exist_ok=True)

# ===========
# Parameters (Top 5 设置)
# ===========
CELLTYPE_COL_CANDS = ["celltypes", "cell_types", "celltype", "cell_type"]
N_WORST_CELLS = 5       # 要绘图的最差细胞数量
N_WORST_PROTEINS = 5    # 要绘图的最差蛋白质数量

# 绘图参数
SCATTER_SIZE = 15
LINE_ALPHA = 0.6
LINE_WIDTH = 1.5
MAX_POINTS_SCATTER = 50000  # overall scatter/contour 时子采样上限，避免点太多

# ===========
# Utilities
# ===========
def to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)

def find_ct_key(adata):
    for k in CELLTYPE_COL_CANDS:
        if k in adata.obs.columns:
            return k
    raise KeyError(f"Cannot find celltype column in adata.obs. Tried {CELLTYPE_COL_CANDS}")

def norm_name(x: str) -> str:
    """统一蛋白命名。"""
    y = re.sub(r'^ADT[-_]*', '', x, flags=re.IGNORECASE)
    y = re.sub(r'(_|-)?TotalSeqB$', '', y, flags=re.IGNORECASE)
    y = re.sub(r'\.protein$', '', y, flags=re.IGNORECASE)
    y = re.sub(r'[^A-Za-z0-9]+', '', y).upper()
    return y

def safe_pcc(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.size == 0 or y.size == 0: return np.nan
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0: return np.nan
    try:
        r, _ = pearsonr(x, y); return r
    except Exception:
        return np.nan

def compute_rmse(y_true, y_pred):
    """计算非NaN值的均方根误差 (Root Mean Squared Error)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(m) == 0: return np.nan
    return np.sqrt(np.mean((y_true[m] - y_pred[m])**2))

def safe_name_for_file(s: str) -> str:
    # 确保模型名称中的括号和空格不会导致路径问题
    return (
        s.replace("/", "_")
         .replace("\\", "_")
         .replace(" ", "_")
         .replace("(", "")
         .replace(")", "")
    )

# ===========
# Alignment for N models
# ===========
def align_many_models(models_triplets):
    """
    加载、对齐所有 AnnData 文件，并返回对齐后的数据和元信息。
    """
    print("Loading AnnData for all models...")
    labels = [t[0] for t in models_triplets]
    gt_list = [ad.read_h5ad(t[1]) for t in models_triplets]
    pr_list = [ad.read_h5ad(t[2]) for t in models_triplets]

    # cells 交集
    common_cells = set(gt_list[0].obs_names)
    for gt, pr in zip(gt_list, pr_list):
        common_cells &= (set(gt.obs_names) & set(pr.obs_names))
    cells = sorted(common_cells)
    if not cells:
        raise ValueError("No common cells across all models.")

    # 蛋白名交集（用规范化名比较）
    v0 = list(map(str, gt_list[0].var_names))
    n0 = [norm_name(s) for s in v0]
    common_norm = set(n0)
    for gt in gt_list[1:]:
        v = list(map(str, gt.var_names))
        n = [norm_name(s) for s in v]
        common_norm &= set(n)
    common = sorted(common_norm)
    if not common:
        raise ValueError("No common proteins across all models after normalization.")
    
    idx0 = {norm_name(v0[i]): i for i in range(len(v0))}
    names_common = [v0[idx0[k]] for k in common]

    # 依次重排/裁剪
    aligned_gt = []
    aligned_pr = []
    for gt, pr in zip(gt_list, pr_list):
        gt = gt[cells].copy()
        pr = pr[cells].copy()

        v = list(map(str, gt.var_names))
        n = [norm_name(s) for s in v]
        idx = {n[i]: i for i in range(len(n))}
        cols = [idx[k] for k in common]

        gt = gt[:, cols].copy()
        pr = pr[:, cols].copy()
        gt.var_names = names_common
        pr.var_names = names_common

        aligned_gt.append(gt)
        aligned_pr.append(pr)

    ct_key = find_ct_key(aligned_gt[0])
    cell_names = aligned_gt[0].obs_names.tolist()
    celltypes = aligned_gt[0].obs[ct_key].astype(str).values
    var_names = list(map(str, aligned_gt[0].var_names))
    gt_list_dense = [to_dense(g.X) for g in aligned_gt]
    pr_list_dense = [to_dense(p.X) for p in aligned_pr]
    
    return cell_names, celltypes, var_names, gt_list_dense, pr_list_dense, labels


# ******************************************************************************
# 核心分析函数：计算指标
# ******************************************************************************

def calculate_all_metrics(Yt_all, Yp_all_list, labels):
    """
    计算所有模型对所有细胞和蛋白质的 PCC 和 RMSE。
    返回: (protein_metrics_list, cell_metrics_list)
    """
    n_cells, n_proteins = Yt_all.shape
    protein_metrics_list = []
    cell_metrics_list = []
    
    for Yp_ref in Yp_all_list:  # 为每个模型独立计算指标
        protein_metrics = []
        for j in range(n_proteins):
            Yt_prot = Yt_all[:, j]
            Yp_prot = Yp_ref[:, j]
            protein_metrics.append({
                'avg_pcc': safe_pcc(Yt_prot, Yp_prot),
                'avg_rmse': compute_rmse(Yt_prot, Yp_prot),
            })

        cell_metrics = []
        for i in range(n_cells):
            Yt_cell = Yt_all[i, :]
            Yp_cell = Yp_ref[i, :]
            cell_metrics.append({
                'avg_pcc': safe_pcc(Yt_cell, Yp_cell),
                'avg_rmse': compute_rmse(Yt_cell, Yp_cell),
            })
        
        protein_metrics_list.append(protein_metrics)
        cell_metrics_list.append(cell_metrics)
        
    return protein_metrics_list, cell_metrics_list


# ******************************************************************************
# 绘图函数
# ******************************************************************************

def plot_worst_cells_scatter(Yt_all, Yp_ref, cell_names, cell_metrics, n_worst, model_label):
    """
    为预测最差的 N 个细胞生成散点图 (Cell-wise Scatter Plot)，
    每个子图为：该细胞所有蛋白的 Ground Truth vs Prediction。
    """
    avg_pccs = np.array([m['avg_pcc'] for m in cell_metrics])
    worst_indices = np.argsort(avg_pccs)[:n_worst]  # 最差 N 个细胞

    print(f"\n## 绘图: {model_label} 最差 {n_worst} 个细胞的散点图 (Cell-wise Scatter)")
    
    n_rows, n_cols = 1, n_worst
    figsize = (3.5 * n_worst, 4.5)
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=170, squeeze=False)
    axes = axes.flatten()
    model_name_short = model_label.split()[0]

    for k, idx in enumerate(worst_indices):
        ax = axes[k]
        xt_cell = Yt_all[idx, :]
        yp_cell = Yp_ref[idx, :]
        
        pcc = safe_pcc(xt_cell, yp_cell)
        rmse = compute_rmse(xt_cell, yp_cell)
        
        ax.scatter(xt_cell, yp_cell, s=SCATTER_SIZE, alpha=LINE_ALPHA, rasterized=True)
        
        min_val = np.nanmin([xt_cell, yp_cell])
        max_val = np.nanmax([xt_cell, yp_cell])
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            min_val, max_val = 0.0, 1.0
        lims = [min_val - 0.1, max_val + 0.1] if max_val > min_val else [min_val - 1, max_val + 1]
        
        ax.plot(lims, lims, 'r--', alpha=0.7, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        cell_name_short = cell_names[idx].split('-')[0]
        ax.set_title(
            f"Cell {k+1} ({cell_name_short})\nPCC({model_name_short}): {pcc:.3f} | RMSE: {rmse:.3f}",
            fontsize=10
        )
        if k == 0:
            ax.set_ylabel("Prediction")
        ax.set_xlabel("Ground Truth")
        ax.grid(alpha=0.2)

    for k in range(len(worst_indices), len(axes)):
        fig.delaxes(axes[k])
        
    plt.tight_layout()
    safe_label = safe_name_for_file(model_label)
    path = os.path.join(OUT_DIR, f"A_{safe_label}_Top{n_worst}_Worst_Cells_Scatter.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_worst_proteins_scatter(Yt_all, Yp_ref, var_names, protein_metrics, n_worst, model_label):
    """
    为预测最差的 N 个蛋白质生成散点图 (Protein-wise Scatter Plot)，
    每个子图为：该蛋白在所有细胞上的 Ground Truth vs Prediction。
    这样 protein 和 cell 都是同一种 GT-vs-Pred 的散点图风格。
    """
    avg_pccs = np.array([m['avg_pcc'] for m in protein_metrics])
    worst_indices = np.argsort(avg_pccs)[:n_worst]  # 最差 N 个蛋白

    print(f"\n## 绘图: {model_label} 最差 {n_worst} 个蛋白质的散点图 (Protein-wise Scatter)")
    
    n_rows, n_cols = 1, n_worst
    figsize = (3.5 * n_worst, 4.5)
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=170, squeeze=False)
    axes = axes.flatten()
    model_name_short = model_label.split()[0]

    for k, idx in enumerate(worst_indices):
        ax = axes[k]
        xt_prot = Yt_all[:, idx]
        yp_prot = Yp_ref[:, idx]

        pcc = safe_pcc(xt_prot, yp_prot)
        rmse = compute_rmse(xt_prot, yp_prot)

        ax.scatter(xt_prot, yp_prot, s=SCATTER_SIZE, alpha=LINE_ALPHA, rasterized=True)

        min_val = np.nanmin([xt_prot, yp_prot])
        max_val = np.nanmax([xt_prot, yp_prot])
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            min_val, max_val = 0.0, 1.0
        lims = [min_val - 0.1, max_val + 0.1] if max_val > min_val else [min_val - 1, max_val + 1]

        ax.plot(lims, lims, 'r--', alpha=0.7, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        prot_name = var_names[idx]
        title = f"{prot_name}\nPCC({model_name_short}): {pcc:.3f} | RMSE: {rmse:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Ground Truth")
        if k == 0:
            ax.set_ylabel("Prediction")
        ax.grid(alpha=0.2)

    for k in range(len(worst_indices), len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    safe_label = safe_name_for_file(model_label)
    path = os.path.join(OUT_DIR, f"B_{safe_label}_Top{n_worst}_Worst_Proteins_Scatter.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_overall_scatter_and_contour(Yt_all, Yp_ref, model_label):
    """
    额外绘制几张整体的 scatter / contour 图：
      1) 所有 cell×protein 打平后的散点图
      2) 所有点的密度 contour 图（带等高线）
    """
    print(f"\n## 绘图: {model_label} Overall Scatter & Contour")

    x = Yt_all.ravel()
    y = Yp_ref.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size == 0:
        print("Warning: no finite data for overall plots.")
        return [],

    # 子采样避免点过多
    if x.size > MAX_POINTS_SCATTER:
        idx = np.random.choice(x.size, MAX_POINTS_SCATTER, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub, y_sub = x, y

    pcc = safe_pcc(x_sub, y_sub)
    rmse = compute_rmse(x_sub, y_sub)
    model_name_short = model_label.split()[0]
    safe_label = safe_name_for_file(model_label)

    paths = []

    # 1) overall scatter
    fig, ax = plt.subplots(figsize=(5, 5), dpi=170)
    ax.scatter(x_sub, y_sub, s=4, alpha=0.4, rasterized=True)

    min_val = np.nanmin([x_sub, y_sub])
    max_val = np.nanmax([x_sub, y_sub])
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        min_val, max_val = 0.0, 1.0
    lims = [min_val - 0.1, max_val + 0.1] if max_val > min_val else [min_val - 1, max_val + 1]

    ax.plot(lims, lims, 'r--', alpha=0.7, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{model_name_short} Overall\nPCC: {pcc:.3f} | RMSE: {rmse:.3f}")
    ax.grid(alpha=0.2)

    path_scatter = os.path.join(OUT_DIR, f"C_{safe_label}_Overall_Scatter.png")
    plt.tight_layout()
    plt.savefig(path_scatter, bbox_inches="tight")
    plt.close()
    paths.append(path_scatter)

    # 2) overall contour (KDE)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=170)

    # KDE 计算
    try:
        values = np.vstack([x_sub, y_sub])
        kde = gaussian_kde(values)
        # 生成网格
        xmin, xmax = lims[0], lims[1]
        ymin, ymax = lims[0], lims[1]
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 100),
            np.linspace(ymin, ymax, 100)
        )
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(grid_coords).reshape(xx.shape)

        # 填色等高线 + 线条等高线
        cf = ax.contourf(xx, yy, zz, levels=15, cmap="viridis")
        cs = ax.contour(xx, yy, zz, levels=15, colors="k", linewidths=0.3, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2g")
        fig.colorbar(cf, ax=ax, label="Density")

    except Exception as e:
        print(f"Contour KDE failed ({e}), using 2D hist instead.")
        # 退化为 2D hist
        cf = ax.hist2d(x_sub, y_sub, bins=60, cmap="viridis")
        fig.colorbar(cf[3], ax=ax, label="Count")

    ax.plot(lims, lims, 'r--', alpha=0.7, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{model_name_short} Overall Density\nPCC: {pcc:.3f} | RMSE: {rmse:.3f}")
    ax.grid(alpha=0.2)

    path_contour = os.path.join(OUT_DIR, f"D_{safe_label}_Overall_Contour.png")
    plt.tight_layout()
    plt.savefig(path_contour, bbox_inches="tight")
    plt.close()
    paths.append(path_contour)

    return paths


# ******************************************************************************
# Main Execution
# ******************************************************************************
def main():
    # 1. 加载并对齐数据
    try:
        cell_names, celltypes, var_names, gt_list, pr_list, labels = align_many_models(MODELS)
    except Exception as e:
        print(f"数据加载或对齐失败: {e}")
        print("请检查 AnnData 文件路径是否正确，或环境中是否安装了 anndata, scanpy, numpy, scipy, matplotlib。")
        return

    Yt_all = gt_list[0]      # Ground Truth（所有模型共用同一份 GT）
    Yp_all_list = pr_list    # List of Predictions

    # 2. 为所有模型分别计算指标
    print("\nCalculating metrics for all models...")
    protein_metrics_list, cell_metrics_list = calculate_all_metrics(Yt_all, Yp_all_list, labels)
    print("Calculation complete.")

    all_paths = []
    
    # 3. 核心循环：遍历所有模型并绘图
    for i, model_label in enumerate(labels):
        Yp_ref = Yp_all_list[i]
        protein_metrics = protein_metrics_list[i]
        cell_metrics = cell_metrics_list[i]

        print(f"\n--- Starting Analysis for {model_label} ---")

        # Plot A: 最差 N 个细胞的 Cell-wise 散点图（和你给的图同风格）
        if N_WORST_CELLS > 0:
            cell_plot_path = plot_worst_cells_scatter(
                Yt_all, Yp_ref, cell_names, cell_metrics,
                N_WORST_CELLS, model_label
            )
            all_paths.append(cell_plot_path)

        # Plot B: 最差 N 个蛋白质的 Protein-wise 散点图（同一种 GT vs Pred 图）
        if N_WORST_PROTEINS > 0:
            protein_plot_path = plot_worst_proteins_scatter(
                Yt_all, Yp_ref, var_names,
                protein_metrics, N_WORST_PROTEINS, model_label
            )
            all_paths.append(protein_plot_path)

        # Plot C & D: 额外的 overall scatter + contour 图
        extra_paths = plot_overall_scatter_and_contour(Yt_all, Yp_ref, model_label)
        all_paths.extend(extra_paths)

    print("\n" + "="*80)
    print("所有模型的 Top N 性能分析图已全部生成。")
    print(f"共生成 {len(all_paths)} 张图。")
    print(f"输出目录: {OUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()