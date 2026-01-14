# compare_multi_models_true_pred.py
import os, re
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.stats import pearsonr
from itertools import islice
import scanpy as sc

# =====================
# Paths & Models
# =====================
# (显示名, groundtruth.h5ad, prediction.h5ad)
MODELS = [
    ("Model-1 (Seurat)",
     "/mnt/scratch/zhan2210/output/Seurat/prediction/SLN-208_8_0_groundtruth.h5ad",
     "/mnt/scratch/zhan2210/output/Seurat/prediction/SLN-208_8_0_prediction.h5ad"),
    ("Model-2 (scmognn)",
     "/mnt/scratch/zhan2210/output/scmognn/dance_SLN-208_8_0_SLN-208_8_0/results/SLN-208_8_0_groundtruth.h5ad",
     "/mnt/scratch/zhan2210/output/scmognn/dance_SLN-208_8_0_SLN-208_8_0/results/SLN-208_8_0_prediction.h5ad"),
    ("Model-3 (sctlgnn)",
     "/mnt/scratch/zhan2210/output/sctlgnn/prediction/SLN-208_8_0_ALL_groundtruth.h5ad",
     "/mnt/scratch/zhan2210/output/sctlgnn/prediction/SLN-208_8_0_ALL_prediction.h5ad"),
]

OUT_DIR  = "/mnt/home/zhan2210/scProtein/code/plot/compare-SLN-208"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# What to plot
# =====================
PLOT_INDIVIDUAL = True     # 每个模型各自单独出图
PLOT_GROUPS     = True     # 按组拼图（多模型叠加）
# 下面这个 GROUPS 既支持“用标签字符串”，也支持“用索引（从0开始）”
GROUPS = [
    # ["Model-1 (Seurat)","Model-2 (scmognn)"], 
    # ["Model-3 (sctlgnn)","Model-4 (hvg2000)"], 
    # ["Model-2 (scmognn)","Model-3 (gasdu-8)"],
    # 你也可以自定义更多组合，比如：
    # [0, 2, 5],  # 用索引选模型
]

# ===========
# Parameters
# ===========
CELLTYPE_COL_CANDS = ["celltypes", "cell_types", "celltype", "cell_type"]
TOPK = 9  # 每个 celltype 选方差最大的 9 个 protein
MAX_CT_OVERVIEW = 6  # 总览图最多显示的 celltype 数

# 颜色（叠加时按顺序使用）
COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

# 透明度（更“不透明” = 数值更大）
# 单独图（single）
ALPHA_SINGLE_ALL = 0.25  # 总览 2x3
ALPHA_SINGLE_TOP = 0.35  # 3x3
# 分组叠加图（group）
ALPHA_GROUP_ALL  = 0.08  # 总览 2x3
ALPHA_GROUP_TOP  = 0.15  # 3x3

# 散点大小
PTS_ALL_SIZE = 4
PTS_TOP_SIZE = 7

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
    """统一蛋白命名：去 ADT-, ADT_, ADT 前缀及常见后缀，保留字母数字并大写。"""
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

def safe_name_for_file(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

def add_identity(ax, xs, *ys_list, margin_ratio=0.02):
    """在坐标上画 y=x 参考线；范围覆盖 true/pred 的数值，略带边距。"""
    arrs = [np.asarray(xs)] + [np.asarray(z) for z in ys_list]
    finite_vals = np.hstack([a[np.isfinite(a)] for a in arrs if a.size > 0])
    if finite_vals.size == 0:
        return
    vmin, vmax = float(np.min(finite_vals)), float(np.max(finite_vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return
    if vmin == vmax:  # 退化
        vmin, vmax = vmin - 1.0, vmax + 1.0
    pad = (vmax - vmin) * margin_ratio
    lo, hi = vmin - pad, vmax + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, alpha=0.7)

# ===========
# Alignment for N models
# ===========
def align_many_models(models_triplets):
    """
    输入: [(label, gt_path, pr_path), ...]
    输出:
      ct_key, celltypes, uniq_ct, var_names,
      gt_list_dense, pr_list_dense, labels
    所有 gt/pr 已对齐到共同 cells & 共同 proteins，列名一致，顺序一致。
    """
    print("Loading AnnData for all models...")
    labels = []
    gt_list = []
    pr_list = []

    # 先读第一组，初始化交集
    first_label, first_gt_path, first_pr_path = models_triplets[0]
    labels.append(first_label)
    gt0 = ad.read_h5ad(first_gt_path)
    pr0 = ad.read_h5ad(first_pr_path)
    gt_list.append(gt0)
    pr_list.append(pr0)

    # cells 交集
    common_cells = set(gt0.obs_names) & set(pr0.obs_names)
    # 蛋白名交集（用规范化名比较）
    v0 = list(map(str, gt0.var_names))
    n0 = [norm_name(s) for s in v0]
    common_norm = set(n0)

    # 其余模型
    for label, gpath, ppath in islice(models_triplets, 1, None):
        labels.append(label)
        gt = ad.read_h5ad(gpath)
        pr = ad.read_h5ad(ppath)
        gt_list.append(gt)
        pr_list.append(pr)

        common_cells &= (set(gt.obs_names) & set(pr.obs_names))
        v = list(map(str, gt.var_names))
        n = [norm_name(s) for s in v]
        common_norm &= set(n)

    # 对齐细胞
    cells = sorted(common_cells)
    if not cells:
        raise ValueError("No common cells across all models.")

    # 对齐蛋白
    common = sorted(common_norm)
    if not common:
        raise ValueError("No common proteins across all models after normalization.")

    # 第一模型的原始名作为展示名
    idx0 = {norm_name(v0[i]): i for i in range(len(v0))}
    names_common = [v0[idx0[k]] for k in common]

    # 依次重排/裁剪到共同 cells & 共同 proteins
    aligned_gt = []
    aligned_pr = []
    for (label, _, _), gt, pr in zip(models_triplets, gt_list, pr_list):
        gt = gt[cells].copy(); pr = pr[cells].copy()

        v = list(map(str, gt.var_names))
        n = [norm_name(s) for s in v]
        idx = {n[i]: i for i in range(len(n))}
        cols = [idx[k] for k in common]

        gt = gt[:, cols].copy()
        pr = pr[:, cols].copy()

        gt.var_names = names_common
        pr.var_names = names_common

        # # --- ( ! ) 在这里添加归一化 ---
        # print(f"Normalizing aligned data for {label}...")
        # sc.pp.normalize_total(gt, target_sum=1e4)
        # sc.pp.log1p(gt)
        # sc.pp.normalize_total(pr, target_sum=1e4)
        # sc.pp.log1p(pr)
        # # --- 归一化结束 ---

        aligned_gt.append(gt)
        aligned_pr.append(pr)

    ct_key = find_ct_key(aligned_gt[0])  # 对齐后 obs 一致
    celltypes = aligned_gt[0].obs[ct_key].astype(str).values
    uniq_ct = sorted(np.unique(celltypes))
    var_names = list(map(str, aligned_gt[0].var_names))

    # 转 dense
    gt_list_dense = [to_dense(g.X) for g in aligned_gt]
    pr_list_dense = [to_dense(p.X) for p in aligned_pr]

    return ct_key, celltypes, uniq_ct, var_names, gt_list_dense, pr_list_dense, labels

# ===========
# Normalizers for groups
# ===========
def _normalize_groups(groups, labels):
    """
    支持用“模型标签字符串”或“索引”来定义分组。
    返回：list[list[int]]（组内是模型索引）
    """
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    normed = []
    for g in groups:
        idxs = []
        for item in g:
            if isinstance(item, int):
                idxs.append(item)
            elif isinstance(item, str):
                if item not in label_to_idx:
                    raise ValueError(f"Group item '{item}' not found in labels: {labels}")
                idxs.append(label_to_idx[item])
            else:
                raise TypeError("GROUPS should contain ints (indices) or strs (labels).")
        # 去重保持顺序
        seen = set(); idxs2 = []
        for x in idxs:
            if x not in seen:
                seen.add(x); idxs2.append(x)
        normed.append(idxs2)
    return normed

# ===========
# Plot helpers
# ===========
def _overview_axes():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=170)
    return fig, np.array(axes).reshape(2, 3)

def _grid_axes():
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), dpi=170)
    return fig, np.array(axes).reshape(3, 3)

def plot_overview_single(uniq_ct, celltypes, Yt_all, Yp, label, color, alpha):
    """单模型总览 2x3 (使用 wise-PCC 指标，GT 为该模型自己的 GT)"""
    fig, axes = _overview_axes()
    for k in range(min(MAX_CT_OVERVIEW, len(uniq_ct))):
        ct = uniq_ct[k]; r, c = divmod(k, 3); ax = axes[r, c]
        mask = (celltypes == ct)
        
        # 1. 获取 2D 子集用于计算指标（该模型自己的 GT & PR）
        Yt_ct = Yt_all[mask, :]
        Yp_ct = Yp[mask, :]
        
        # 2. 获取扁平化数据用于绘制散点
        xt_flat = Yt_ct.ravel()
        yp_flat = Yp_ct.ravel()

        add_identity(ax, xt_flat, yp_flat)
        m = np.isfinite(xt_flat) & np.isfinite(yp_flat)
        ax.scatter(xt_flat[m], yp_flat[m], s=PTS_ALL_SIZE, alpha=alpha,
                   color=color, label=label, rasterized=True)
        
        # 3. 计算 wise-PCC 指标（每个模型用自己的 GT）
        pcc_prot, pcc_cell = compute_wise_pcc(Yt_ct, Yp_ct)
        
        # 4. 更新标题
        ax.set_title(f"{ct} ({label.split()[0]})\nProt-PCC: {pcc_prot:.3f} | Cell-PCC: {pcc_cell:.3f}")
        ax.set_xlabel("true"); ax.set_ylabel("pred")
        
    handles, legends = axes[0,0].get_legend_handles_labels()
    if handles: axes[0,0].legend(frameon=False, fontsize=9, loc="upper left")
    for k in range(min(MAX_CT_OVERVIEW, len(uniq_ct)), 6):
        r, c = divmod(k, 3); axes[r, c].axis("off")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"overall_2x3_single_{safe_name_for_file(label)}.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved overview single: {path}")

def plot_grid_single(ct, mask, var_names, Yt_all, Yp, label, color, alpha):
    """单模型 3x3（该模型自己的 GT）"""
    Yt_ct = Yt_all[mask, :]
    vari = np.var(Yt_ct, axis=0)
    idx = np.argsort(vari)[::-1][:TOPK]
    names = [var_names[j] for j in idx]
    fig, axes = _grid_axes()
    for t, (j, nm) in enumerate(zip(idx, names)):
        r, c = divmod(t, 3); ax = axes[r, c]
        x_true = Yt_ct[:, j]; y = Yp[mask, j]
        add_identity(ax, x_true, y)
        m = np.isfinite(x_true) & np.isfinite(y)
        ax.scatter(x_true[m], y[m], s=PTS_TOP_SIZE, alpha=alpha,
                   color=color, label=label, rasterized=True)
        pcc = safe_pcc(x_true, y)
        ax.set_title(f"{nm}\nPCC {label.split()[0]}: {pcc:.2f}", fontsize=10)
        ax.set_xlabel("true"); ax.set_ylabel("pred")
    handles, legends = axes[0,0].get_legend_handles_labels()
    if handles: axes[0,0].legend(frameon=False, fontsize=8, loc="upper left")
    for k in range(TOPK, 9):
        r, c = divmod(k, 3); axes[r, c].axis("off")
    plt.tight_layout()
    safe_ct = safe_name_for_file(ct)
    out_path = os.path.join(OUT_DIR, f"{safe_ct}_top{TOPK}_3x3_single_{safe_name_for_file(label)}.png")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()
    # print(f"[{ct}] single saved: {out_path}")

def plot_overview_group(uniq_ct, celltypes, Yt_list, Yp_list, labels, color_cycle, alpha):
    """
    多模型叠加 总览 2x3（按组, 使用 wise-PCC 指标）
    每个模型使用自己的 GT: Yt_list[i] 搭配 Yp_list[i]
    """
    fig, axes = _overview_axes()
    for k in range(min(MAX_CT_OVERVIEW, len(uniq_ct))):
        ct = uniq_ct[k]; r, c = divmod(k, 3); ax = axes[r, c]
        mask = (celltypes == ct)
        
        # 1. 获取 2D 子集（每个模型自己的 GT & PR）
        Yt_ct_list = [Yt[mask, :] for Yt in Yt_list]
        Yp_ct_list = [Yp[mask, :] for Yp in Yp_list]

        # 用第一个模型的数据来决定 identity 线的范围（只是视觉参考）
        xt_flat_ref = Yt_ct_list[0].ravel()
        yp_flat_ref_list = [Yp_ct.ravel() for Yp_ct in Yp_ct_list]
        add_identity(ax, xt_flat_ref, *yp_flat_ref_list)
        
        # 2. 循环计算指标并绘图（每个模型自己的 GT）
        metric_texts = []
        for i, (Yt_ct, Yp_ct) in enumerate(zip(Yt_ct_list, Yp_ct_list)):
            xt_flat = Yt_ct.ravel()
            yp_flat = Yp_ct.ravel()
            m = np.isfinite(xt_flat) & np.isfinite(yp_flat)
            ax.scatter(
                xt_flat[m], yp_flat[m],
                s=PTS_ALL_SIZE,
                alpha=alpha,
                color=color_cycle[i % len(color_cycle)],
                label=labels[i],
                rasterized=True
            )
            pcc_prot, pcc_cell = compute_wise_pcc(Yt_ct, Yp_ct)
            metric_texts.append(f"{labels[i].split()[0]} (P:{pcc_prot:.3f}, C:{pcc_cell:.3f})")

        ax.set_title(f"{ct}\n{' | '.join(metric_texts)}")
        ax.set_xlabel("true"); ax.set_ylabel("pred")
        
    handles, legends = axes[0,0].get_legend_handles_labels()
    if handles: axes[0,0].legend(frameon=False, fontsize=9, loc="upper left")
    for k in range(min(MAX_CT_OVERVIEW, len(uniq_ct)), 6):
        r, c = divmod(k, 3); axes[r, c].axis("off")
    plt.tight_layout()
    name = "_AND_".join([safe_name_for_file(l) for l in labels])
    path = os.path.join(OUT_DIR, f"overall_2x3_group_{name}.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved overview group: {path}")

def plot_grid_group(ct, mask, var_names, Yt_list, Yp_list, labels, color_cycle, alpha):
    """
    多模型叠加 3x3（按组）
    - TopK 的 protein 选择仍然基于第一个模型的 GT 方差
    - 每个模型自己的 GT 与自己的 PR 做散点和 PCC
    """
    # 用第一个模型来选 topK 蛋白
    Yt_ct_ref = Yt_list[0][mask, :]
    vari = np.var(Yt_ct_ref, axis=0)
    idx = np.argsort(vari)[::-1][:TOPK]
    names = [var_names[j] for j in idx]

    fig, axes = _grid_axes()
    for t, (j, nm) in enumerate(zip(idx, names)):
        r, c = divmod(t, 3); ax = axes[r, c]

        # identity 线用第一个模型数据设范围
        x_true_ref = Yt_ct_ref[:, j]
        y_models_ref = [Yp[mask, j] for Yp in Yp_list]
        add_identity(ax, x_true_ref, *y_models_ref)

        pccs = []
        for i, (Yt_all, Yp_all) in enumerate(zip(Yt_list, Yp_list)):
            x_true = Yt_all[mask, j]
            y = Yp_all[mask, j]
            m = np.isfinite(x_true) & np.isfinite(y)
            ax.scatter(
                x_true[m], y[m],
                s=PTS_TOP_SIZE,
                alpha=alpha,
                color=color_cycle[i % len(color_cycle)],
                label=labels[i],
                rasterized=True
            )
            pccs.append(safe_pcc(x_true, y))

        pcc_text = " | ".join([f"{labels[i].split()[0]}: {pccs[i]:.2f}" for i in range(len(labels))])
        ax.set_title(f"{nm}\nPCC {pcc_text}", fontsize=10)
        ax.set_xlabel("true"); ax.set_ylabel("pred")

    handles, legends = axes[0,0].get_legend_handles_labels()
    if handles: axes[0,0].legend(frameon=False, fontsize=8, loc="upper left")
    for k in range(TOPK, 9):
        r, c = divmod(k, 3); axes[r, c].axis("off")
    plt.tight_layout()
    safe_ct = safe_name_for_file(ct)
    name = "_AND_".join([safe_name_for_file(l) for l in labels])
    out_path = os.path.join(OUT_DIR, f"{safe_ct}_top{TOPK}_3x3_group_{name}.png")
    plt.savefig(out_path, bbox_inches="tight"); plt.close()
    # print(f"[{ct}] group saved: {out_path}")
    
def compute_wise_pcc(y_true, y_pred):
    """
    计算 cell-wise 和 protein-wise 的平均 PCC。
    假定 y_true 和 y_pred 具有相同的 shape [n_cells, n_proteins]
    
    返回: (mean_protein_pcc, mean_cell_pcc)
    """
    if y_true.size == 0 or y_pred.size == 0 or y_true.shape != y_pred.shape:
        return np.nan, np.nan
        
    # Protein-wise (沿 axis=0 计算，即逐个 column)
    protein_pccs = []
    for j in range(y_true.shape[1]): # 遍历所有 protein
        col_true = y_true[:, j]
        col_pred = y_pred[:, j]
        protein_pccs.append(safe_pcc(col_true, col_pred))
    
    # Cell-wise (沿 axis=1 计算，即逐个 row)
    cell_pccs = []
    for i in range(y_true.shape[0]): # 遍历所有 cell
        row_true = y_true[i, :]
        row_pred = y_pred[i, :]
        cell_pccs.append(safe_pcc(row_true, row_pred))
        
    mean_prot_pcc = np.nanmean(protein_pccs)
    mean_cell_pcc = np.nanmean(cell_pccs)
    
    return mean_prot_pcc, mean_cell_pcc

# ===========
# Main
# ===========
def main():
    ct_key, celltypes, uniq_ct, var_names, gt_list, pr_list, labels = align_many_models(MODELS)

    # 每个模型用自己的 GT
    Yt_all_list = gt_list      # list of [n_cells, n_proteins]
    Yp_all_list = pr_list      # list of [n_cells, n_proteins]
    n_ct = min(MAX_CT_OVERVIEW, len(uniq_ct))

    # ---- 1) 每个模型各出一张（自用 GT）----
    if PLOT_INDIVIDUAL:
        print("Plotting single-model figures (each model with its own GT)...")
        for i, lab in enumerate(labels):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            Yt_i = Yt_all_list[i]
            Yp_i = Yp_all_list[i]
            # 总览 2x3
            plot_overview_single(uniq_ct, celltypes, Yt_i, Yp_i, lab, color, ALPHA_SINGLE_ALL)
            # 每个 celltype 3x3
            for ct in uniq_ct[:n_ct]:
                mask = (celltypes == ct)
                plot_grid_single(ct, mask, var_names, Yt_i, Yp_i, lab, color, ALPHA_SINGLE_TOP)

    # ---- 2) 分组叠加图（每个模型用自己的 GT）----
    if PLOT_GROUPS and GROUPS:
        print("Plotting grouped figures (each model with its own GT)...")
        group_idxs_list = _normalize_groups(GROUPS, labels)  # 统一转成索引
        for gidxs in group_idxs_list:
            grp_labels = [labels[j] for j in gidxs]
            grp_Yts   = [Yt_all_list[j] for j in gidxs]
            grp_Yps   = [Yp_all_list[j] for j in gidxs]
            grp_colors= [COLOR_CYCLE[j % len(COLOR_CYCLE)] for j in range(len(gidxs))]
            # 总览 2x3
            plot_overview_group(uniq_ct, celltypes, grp_Yts, grp_Yps, grp_labels, grp_colors, ALPHA_GROUP_ALL)
            # 每个 celltype 3x3
            for ct in uniq_ct[:n_ct]:
                mask = (celltypes == ct)
                plot_grid_group(ct, mask, var_names, grp_Yts, grp_Yps, grp_labels, grp_colors, ALPHA_GROUP_TOP)

    print("Done.")

if __name__ == "__main__":
    main()