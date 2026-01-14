# plot_all_celltypes.py — compare models on every celltype
# Violin plots + two overview plots + radar plots (axis-zoomed, no value change for PCC)
# Heatmaps (4) + Metric CSVs (4): mean(4dp) & variance(sci)
# Diagnostics scatter plots for all cells & proteins (PCC vs RMSE), colored by celltype
# UPDATED: Per-celltype scatter uses raw rows (model×repi×cell), overlap as bubble size; removed PDF & full heatmaps

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colormaps

# ===================================================================
# 1) Config
# ===================================================================
MODEL_PATHS = {
    # "softplus":         ("/mnt/scratch/zhan2210/output/softplus/results/", "PBMC-Li"),
    "gasdu-15":         ("/mnt/scratch/zhan2210/output/gasdu-15/results/", "SLN-208"),
    "gasdu-8":         ("/mnt/scratch/zhan2210/output/gasdu-8/results/", "SLN-208"),
    "seurat":           ("/mnt/scratch/zhan2210/output/Seurat/results/", "SLN-208"),
}
REPI_RANGE = range(10)

ROOT_OUTPUT = "/mnt/home/zhan2210/scProtein/code/SLN-208-plot"
OUTPUT_DIR = ROOT_OUTPUT
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="ticks", context="talk")


# ===================================================================
# 2) Load & Tidy
# ===================================================================
def load_all_results(model_paths: dict, repi_range) -> pd.DataFrame:
    """
    Load protein/cell level metrics from multiple model dirs and merge
    into a long table: ['model','level','celltype','metric','value','repi','row_id','protein'(opt)].
    'row_id' uniquely identifies each row within each source file, so that PCC/RMSE
    can be paired without collapsing multiple cells/proteins.
    """
    frames = []
    print("Starting to load data...")
    for model_name, base_path in model_paths.items():
        print(f"-> Processing model: {model_name}")
        for repi in repi_range:
            base_path, prefix = model_paths[model_name]

            if model_name == "seurat":
                protein_file = os.path.join(base_path, f"{prefix}_{repi}_0_protein_metrics.csv")
                cell_file    = os.path.join(base_path, f"{prefix}_{repi}_0_cell_metrics.csv")
            else:
                protein_file = os.path.join(base_path, f"{prefix}_{repi}_0_ALL_protein_metrics.csv")
                cell_file    = os.path.join(base_path, f"{prefix}_{repi}_0_ALL_cell_metrics.csv")
            # ===============================================================
            # protein level
            try:
                dfp = pd.read_csv(protein_file)
                if "celltype" not in dfp.columns:
                    raise KeyError(f"'celltype' column missing in {protein_file}")
                dfp = dfp.reset_index().rename(columns={"index": "row_infile"})
                dfp["row_id"] = dfp["row_infile"].astype(str).radd(f"{prefix}_{repi}_protein_")
                keep_cols = ["pcc", "rmse", "model", "level", "celltype", "repi", "row_id"]
                dfp["model"] = model_name
                dfp["level"] = "Protein"
                dfp["repi"]  = repi
                if "protein" in dfp.columns:
                    keep_cols.append("protein")
                frames.append(dfp[keep_cols])
            except FileNotFoundError:
                print(f"   - Warning: Protein file not found, skipping: {protein_file}")

            # cell level
            try:
                dfc = pd.read_csv(cell_file)
                if "celltype" not in dfc.columns:
                    raise KeyError(f"'celltype' column missing in {cell_file}")
                dfc = dfc.reset_index().rename(columns={"index": "row_infile"})
                dfc["row_id"] = dfc["row_infile"].astype(str).radd(f"{prefix}_{repi}_cell_")
                keep_cols = ["pcc", "rmse", "model", "level", "celltype", "repi", "row_id"]
                dfc["model"] = model_name
                dfc["level"] = "Cell"
                dfc["repi"]  = repi
                frames.append(dfc[keep_cols])
            except FileNotFoundError:
                print(f"   - Warning: Cell file not found, skipping: {cell_file}")

    if not frames:
        raise RuntimeError("No data loaded. Check MODEL_PATHS or file availability.")

    df = pd.concat(frames, ignore_index=True)

    # Long-form with metric in ['PCC','RMSE']
    id_vars = ["model", "level", "celltype", "repi", "row_id"]
    if "protein" in df.columns:
        id_vars.append("protein")

    df_tidy = df.melt(
        id_vars=id_vars,
        value_vars=["pcc", "rmse"],
        var_name="metric",
        value_name="value"
    )
    df_tidy["metric"] = df_tidy["metric"].str.upper()

    # Preserve model order
    model_order = list(model_paths.keys())
    df_tidy["model"] = pd.Categorical(df_tidy["model"], categories=model_order, ordered=True)

    # 统一 celltype 字符串空白
    df_tidy["celltype"] = df_tidy["celltype"].astype(str).str.strip()

    print("\nData loading and processing complete.")
    print("Total data points loaded:", len(df_tidy))
    return df_tidy


# ===================================================================
# 3) Base plotting helpers (ALL VIOLIN)
# ===================================================================
def create_violinplot(ax, data, level, metric, model_order):
    plot_data = data.query("level == @level and metric == @metric")
    if plot_data.empty:
        ax.text(0.5, 0.5, f"No data for {level}-{metric}", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel(metric); ax.set_ylabel(""); ax.set_title(level)
        return
    sns.violinplot(ax=ax, data=plot_data, x="value", y="model",
                   order=model_order, orient="h", inner="quartile", cut=0)
    ax.set_title(level); ax.set_xlabel(metric); ax.set_ylabel("")


def plot_for_celltype(df, ct_label, out_path, model_order):
    df_ct = df.query("celltype == @ct_label").copy()
    if df_ct.empty:
        print(f"[Warn] No rows for celltype={ct_label}, skip plotting."); return
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    create_violinplot(axes[0, 0], df_ct, "Protein", "PCC", model_order)
    create_violinplot(axes[0, 1], df_ct, "Protein", "RMSE", model_order)
    create_violinplot(axes[1, 0], df_ct, "Cell",    "PCC", model_order)
    create_violinplot(axes[1, 1], df_ct, "Cell",    "RMSE", model_order)
    fig.suptitle(f"Model comparison — celltype: {ct_label}", fontsize=16)
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"✓ Saved: {out_path}")


def plot_overview_all_celltypes(df, out_path_png, model_order):
    df_use = df[df["celltype"] != "ALL"].copy()
    if df_use.empty:
        print("[Warn] No real celltype rows (excluding ALL). Skip overview."); return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=True)
    panels = [
        ("Protein", "PCC",  axes[0, 0]),
        ("Protein", "RMSE", axes[0, 1]),
        ("Cell",    "PCC",  axes[1, 0]),
        ("Cell",    "RMSE", axes[1, 1]),
    ]
    for level, metric, ax in panels:
        plot_data = df_use.query("level == @level and metric == @metric")
        if plot_data.empty:
            ax.text(0.5, 0.5, f"No data for {level}-{metric}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{level} — {metric}"); ax.set_xlabel(metric); ax.set_ylabel(""); continue
        sns.violinplot(ax=ax, data=plot_data, x="value", y="model",
                       order=model_order, orient="h", inner="quartile", cut=0)
        ax.set_title(f"{level} — {metric}")
        ax.set_xlabel(metric); ax.set_ylabel("")
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"✓ Saved overview: {out_path_png}")


# ===================================================================
# 4) OVERVIEW (A): Overlayed violins colored by celltype
# ===================================================================
def overlay_violin_by_celltype(df, out_path_png, model_order, max_celltypes=None):
    df_use = df[df["celltype"] != "ALL"].copy()
    if df_use.empty:
        print("[Warn] No real celltype rows (excluding ALL). Skip overlay overview."); return

    celltypes = sorted(df_use["celltype"].unique())
    if max_celltypes is not None:
        celltypes = celltypes[:max_celltypes]

    cmap = colormaps.get_cmap("tab20")
    colors = {ct: cmap(i % cmap.N) for i, ct in enumerate(celltypes)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=True)
    panels = [
        ("Protein", "PCC",  axes[0, 0]),
        ("Protein", "RMSE", axes[0, 1]),
        ("Cell",    "PCC",  axes[1, 0]),
        ("Cell",    "RMSE", axes[1, 1]),
    ]

    for level, metric, ax in panels:
        subset = df_use.query("level == @level and metric == @metric")
        if subset.empty:
            ax.text(0.5, 0.5, f"No data for {level}-{metric}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{level} — {metric}"); ax.set_xlabel(metric); ax.set_ylabel(""); continue

        # Base (to set y-order)
        sns.violinplot(ax=ax, data=subset, x="value", y="model",
                       order=model_order, orient="h", inner=None, cut=0, color="white")

        # Overlay each celltype
        for ct in celltypes:
            ct_data = subset[subset["celltype"] == ct]
            if ct_data.empty: continue
            sns.violinplot(ax=ax, data=ct_data, x="value", y="model",
                           order=model_order, orient="h", inner=None, cut=0,
                           color=colors[ct], linewidth=0, alpha=0.25)

        ax.set_title(f"{level} — {metric}")
        ax.set_xlabel(metric); ax.set_ylabel("")

    handles = [plt.Line2D([0], [0], color=colors[ct], lw=10, alpha=0.6) for ct in celltypes]
    fig.legend(handles, celltypes, loc="upper center", ncols=5, title="Celltype", frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.93], pad=2.0)
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"✓ Saved overlay-by-celltype overview: {out_path_png}")


# ===================================================================
# 5) Radar plots (axis zooming as requested)
# ===================================================================
def _library_size_weighted_means(df: pd.DataFrame) -> pd.DataFrame:
    per_repi = (df.groupby(["model", "repi", "level", "metric", "celltype"], observed=True)["value"]
                  .agg(value_mean="mean", n="size").reset_index())

    def _wmean(g):
        v = g["value_mean"].astype(float).to_numpy()
        w = g["n"].astype(float).to_numpy()
        if np.all(w == 0):
            return np.nan
        return np.sum(v * w) / np.sum(w)

    wavg = (per_repi
            .groupby(["model", "level", "metric", "celltype"], observed=True)
            .apply(_wmean, include_groups=False)
            .reset_index(name="value_wmean"))
    return wavg


def _axis_window(vmin, vmax, pad_frac=0.1, hard_min=None, hard_max=None):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if math.isclose(vmin, vmax):
        rng = max(1e-3, abs(vmin) * 1e-3 + 1e-3)
    else:
        rng = vmax - vmin
    rmin = vmin - pad_frac * rng
    rmax = vmax + pad_frac * rng
    if hard_min is not None:
        rmin = max(hard_min, rmin)
    if hard_max is not None:
        rmax = min(hard_max, rmax)
    if rmin >= rmax:
        rmin, rmax = vmin - 1e-3, vmax + 1e-3
    return rmin, rmax


def plot_radar_by_metric_models(df, out_dir, model_order, max_celltypes=None, alpha=0.25):
    os.makedirs(out_dir, exist_ok=True)
    wavg = _library_size_weighted_means(df)

    celltypes_all = sorted([ct for ct in wavg["celltype"].unique() if ct != "ALL"])
    if max_celltypes is not None:
        celltypes_all = celltypes_all[:max_celltypes]

    cmap = colormaps.get_cmap("tab10")
    model_list = list(model_order)
    model_colors = {m: cmap(i % cmap.N) for i, m in enumerate(model_list)}

    blocks = [
        ("Protein", "PCC"),
        ("Protein", "RMSE"),
        ("Cell", "PCC"),
        ("Cell", "RMSE"),
    ]

    for level, metric in blocks:
        block = wavg[(wavg["level"] == level) & (wavg["metric"] == metric)].copy()
        mat = block.pivot_table(index="model", columns="celltype", values="value_wmean")
        mat = mat.reindex(index=model_list, columns=celltypes_all)

        if mat.empty or mat.values.size == 0:
            print(f"[Warn] No celltypes for radar: {level}-{metric}"); continue

        vals = mat.values.astype(float)

        if metric.upper() == "PCC":
            display_vals = vals.copy()
            disp_min, disp_max = np.nanmin(display_vals), np.nanmax(display_vals)
            rmin, rmax = _axis_window(disp_min, disp_max, pad_frac=0.1, hard_min=-1.0, hard_max=1.0)
            r_ticks = np.linspace(rmin, rmax, 5)
            title_suffix = f" (PCC raw; axis zoom [{rmin:.3f},{rmax:.3f}])"
        else:
            display_vals = np.log10(1.0 + np.clip(vals, a_min=0, a_max=None))
            disp_min, disp_max = np.nanmin(display_vals), np.nanmax(display_vals)
            rmin, rmax = _axis_window(disp_min, disp_max, pad_frac=0.1)
            r_ticks = np.linspace(rmin, rmax, 5)
            title_suffix = " (display=log10(1+RMSE); axis zoom)"

        # Radar setup
        N = len(celltypes_all)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(figsize=(12, 12))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels=celltypes_all, fontsize=9)

        ax.set_ylim(rmin, rmax)
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f"{t:.3f}" for t in r_ticks])
        ax.set_title(f"Radar — {level}-{metric}{title_suffix}", va='bottom')

        disp_df = pd.DataFrame(display_vals, index=mat.index, columns=mat.columns)
        for m in model_list:
            if m not in disp_df.index:
                continue
            series = disp_df.loc[m]
            if series.isna().all():
                continue
            values = series.tolist()
            if any([v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) for v in values]):
                continue
            values += values[:1]
            ax.plot(angles, values, color=model_colors[m], linewidth=2, alpha=0.95, label=str(m))
            ax.fill(angles, values, color=model_colors[m], alpha=alpha)

        leg = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05),
                        title="Model", frameon=False, ncol=1)
        for lh in leg.legend_handles:
            lh.set_alpha(0.95)

        fname = f"radar_{level}_{metric}.png".replace(" ", "")
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"✓ Saved radar (axis-zoomed): {out_path}")


# ===================================================================
# 6) Experiment 0_0 summary (heatmaps for quick glance)
# ===================================================================
def plot_experiment_00_summary(df, out_dir, model_order):
    os.makedirs(out_dir, exist_ok=True)
    d0 = df[(df["repi"] == 0) & (df["celltype"] != "ALL")].copy()
    if d0.empty:
        print("[Warn] No rows for repi=0."); return

    g = (d0.groupby(["model","level","metric","celltype"], observed=True)["value"]
            .mean().reset_index())

    blocks = [
        ("Protein", "PCC"),
        ("Protein", "RMSE"),
        ("Cell", "PCC"),
        ("Cell", "RMSE"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for (level, metric), ax in zip(blocks, axes.ravel()):
        sub = g[(g["level"] == level) & (g["metric"] == metric)].copy()
        if sub.empty:
            ax.axis("off"); ax.set_title(f"{level}-{metric} (no data)"); continue
        mat = sub.pivot_table(index="celltype", columns="model", values="value")
        if metric == "PCC":
            mat = mat.reindex(index=mat.mean(axis=1).sort_values(ascending=False).index)
        else:
            mat = mat.reindex(index=mat.mean(axis=1).sort_values(ascending=True).index)
        mat = mat.reindex(columns=model_order)

        sns.heatmap(mat, ax=ax, cmap="viridis", cbar=True)
        ax.set_title(f"{level}-{metric} (repi=0)")
        ax.set_xlabel("model"); ax.set_ylabel("celltype")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    outp = os.path.join(out_dir, "exp_0_0_heatmaps.png")
    plt.savefig(outp, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"✓ Saved 0_0 heatmaps: {outp}")


# ===================================================================
# 7) 诊断散点图：所有 cell / protein 的 PCC vs RMSE（标注 celltype）
# ===================================================================
def plot_diagnostics_scatters(df, out_dir, model_order):
    os.makedirs(out_dir, exist_ok=True)

    id_cols = ["model", "repi", "level", "celltype", "row_id"]
    id_cols_protein = id_cols + ["protein"] if "protein" in df.columns else None

    def _pair_table(dfin, level_name, id_cols_used):
        sub = dfin[dfin["level"] == level_name]
        wide = sub.pivot_table(index=id_cols_used, columns="metric", values="value", aggfunc="first").reset_index()
        for m in ["PCC", "RMSE"]:
            if m not in wide.columns:
                wide[m] = np.nan
        return wide

    # Cell level
    cell_pairs = _pair_table(df, "Cell", id_cols)
    if not cell_pairs.empty:
        plt.figure(figsize=(12, 9))
        markers = ["o", "s", "^"]
        model_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(model_order)}
        ct_list = sorted(cell_pairs["celltype"].dropna().unique())
        cmap = colormaps.get_cmap("tab20")
        ct_to_color = {ct: cmap(i % cmap.N) for i, ct in enumerate(ct_list)}

        for m in model_order:
            dfm = cell_pairs[cell_pairs["model"] == m]
            if dfm.empty: continue
            plt.scatter(
                dfm["PCC"], dfm["RMSE"],
                label=m,
                marker=model_to_marker[m],
                s=12,
                edgecolors="none",
                c=[ct_to_color.get(ct, (0.5, 0.5, 0.5, 1.0)) for ct in dfm["celltype"]]
            )

        from matplotlib.lines import Line2D
        handles_ct = [Line2D([0], [0], marker="o", color='w', label=ct,
                              markerfacecolor=ct_to_color[ct], markersize=6)
                      for ct in ct_list[:20]]
        plt.legend(title="Model", loc="upper right")
        if len(handles_ct):
            plt.gca().add_artist(plt.legend(handles=handles_ct, title="Celltype (sampled)", loc="lower left", frameon=False))

        plt.xlabel("PCC (raw)")
        plt.ylabel("RMSE (raw)")
        plt.title("All Cells — PCC vs RMSE (color=celltype, marker=model)")
        outp = os.path.join(out_dir, "all_cells_scatter.png")
        plt.tight_layout(); plt.savefig(outp, dpi=300, bbox_inches="tight"); plt.close()
        print(f"✓ Saved diagnostics scatter (cells): {outp}")

    # Protein level (if any)
    if id_cols_protein is not None:
        prot_pairs = _pair_table(df, "Protein", id_cols_protein)
        if not prot_pairs.empty:
            plt.figure(figsize=(12, 9))
            markers = ["o", "s", "^"]
            model_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(model_order)}
            ct_list = sorted(prot_pairs["celltype"].dropna().unique())
            cmap = colormaps.get_cmap("tab20")
            ct_to_color = {ct: cmap(i % cmap.N) for i, ct in enumerate(ct_list)}

            for m in model_order:
                dfm = prot_pairs[prot_pairs["model"] == m]
                if dfm.empty: continue
                plt.scatter(
                    dfm["PCC"], dfm["RMSE"],
                    label=m,
                    marker=model_to_marker[m],
                    s=8,
                    edgecolors="none",
                    c=[ct_to_color.get(ct, (0.5, 0.5, 0.5, 1.0)) for ct in dfm["celltype"]]
                )

            from matplotlib.lines import Line2D
            handles_ct = [Line2D([0], [0], marker="o", color='w', label=ct,
                                  markerfacecolor=ct_to_color[ct], markersize=6)
                          for ct in ct_list[:20]]
            plt.legend(title="Model", loc="upper right")
            if len(handles_ct):
                plt.gca().add_artist(plt.legend(handles=handles_ct, title="Celltype (sampled)", loc="lower left", frameon=False))

            plt.xlabel("PCC (raw)")
            plt.ylabel("RMSE (raw)")
            plt.title("All Proteins — PCC vs RMSE (color=celltype, marker=model)")
            outp = os.path.join(out_dir, "all_proteins_scatter.png")
            plt.tight_layout(); plt.savefig(outp, dpi=300, bbox_inches="tight"); plt.close()
            print(f"✓ Saved diagnostics scatter (proteins): {outp}")


# ===================================================================
# 8) 每个 celltype 单独一张散点图（Cell 层，颜色=模型；重合=更大的气泡）
# ===================================================================
def plot_scatter_by_celltype_cells(df, out_dir, model_order):
    """
    One figure per celltype:
      - Data points: each raw row (model, repi, celltype, row_id) with paired PCC & RMSE
      - Color = model; bubble size encodes overlap frequency of identical (PCC,RMSE)
      - No averaging and no counts shown in title
    """
    out_dir = os.path.join(out_dir, "per_celltype_scatter")
    os.makedirs(out_dir, exist_ok=True)

    # pair PCC/RMSE on raw rows
    sub = df[(df["level"] == "Cell") & (df["celltype"] != "ALL")].copy()
    wide = sub.pivot_table(index=["celltype", "model", "repi", "row_id"],
                           columns="metric", values="value", aggfunc="first").reset_index()
    if wide.empty:
        print("[Warn] No cell-level data for per-celltype scatter."); return

    # colors by model
    cmap = colormaps.get_cmap("tab10")
    models = list(model_order)
    model_colors = {m: cmap(i % cmap.N) for i, m in enumerate(models)}

    for ct in sorted(wide["celltype"].unique()):
        wct = wide[wide["celltype"] == ct]
        if wct.empty:
            continue
        plt.figure(figsize=(7.5, 6.0))
        for m in models:
            wm = wct[wct["model"] == m]
            if wm.empty:
                continue
            # 合并完全相同的 (PCC, RMSE) 点，计频次 -> 点大小
            grouped = (wm.groupby(["PCC", "RMSE"])
                         .size().reset_index(name="freq"))
            sizes = 18 + 14 * np.sqrt(grouped["freq"].values)  # 频次对应气泡半径
            plt.scatter(grouped["PCC"], grouped["RMSE"], label=m,
                        c=[model_colors[m]], s=sizes, alpha=0.75, edgecolors="none")

        # 紧凑坐标轴
        def _tight_axis(arr, pad=0.05):
            arr = np.asarray(arr, dtype=float); arr = arr[np.isfinite(arr)]
            if arr.size == 0: return (0.0, 1.0)
            lo, hi = arr.min(), arr.max()
            if math.isclose(lo, hi):
                rng = max(1e-3, abs(lo)*1e-3 + 1e-3); return lo-rng, hi+rng
            rng = hi - lo; return lo - pad*rng, hi + pad*rng

        xlo, xhi = _tight_axis(wct["PCC"])
        ylo, yhi = _tight_axis(wct["RMSE"])
        plt.xlim(max(-1.0, xlo), min(1.0, xhi))
        plt.ylim(ylo, yhi)

        plt.xlabel("PCC (raw)")
        plt.ylabel("RMSE (raw)")
        plt.title(f"{ct} — PCC vs RMSE (color=model, size=overlap)")
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()

        safe = str(ct).replace(" ", "_").replace("/", "_").replace("\\", "_")
        outp = os.path.join(out_dir, f"scatter_celltype_{safe}.png")
        plt.savefig(outp, dpi=300, bbox_inches="tight"); plt.close()
        print(f"✓ Saved per-celltype scatter: {outp}")


# ===================================================================
# 9) 四个 CSV：各指标 (mean 4dp, var scientific), 列为 celltype_*, 末尾 ALL_*
# ===================================================================
def _format_mean(x):
    if pd.isna(x): return ""
    return f"{x:.4f}"

def _format_var(x):
    if pd.isna(x): return ""
    return f"{x:.3e}"

def save_metric_csvs(df, out_dir, model_order):
    os.makedirs(out_dir, exist_ok=True)
    df_use = df[df["celltype"] != "ALL"].copy()
    if df_use.empty:
        print("[Warn] No real celltype rows (excluding ALL). Skip metric CSVs."); return

    blocks = [
        ("Protein", "PCC"),
        ("Protein", "RMSE"),
        ("Cell", "PCC"),
        ("Cell", "RMSE"),
    ]

    celltypes = sorted(df_use["celltype"].unique())
    models = list(model_order)

    for level, metric in blocks:
        sub = df_use[(df_use["level"] == level) & (df_use["metric"] == metric)].copy()
        if sub.empty:
            print(f"[Warn] No data for {level}-{metric}, skip CSV."); continue

        agg = (sub.groupby(["model","celltype"], observed=True)["value"]
                  .agg(value_mean="mean", value_var="var").reset_index())

        mean_wide = agg.pivot_table(index="model", columns="celltype", values="value_mean").reindex(index=models, columns=celltypes)
        var_wide  = agg.pivot_table(index="model", columns="celltype", values="value_var").reindex(index=models, columns=celltypes)

        all_mean = mean_wide.mean(axis=1, skipna=True)
        all_var  = var_wide.mean(axis=1, skipna=True)

        data_frames = []
        for ct in celltypes:
            df_pair = pd.DataFrame({
                f"{ct}_mean": mean_wide[ct],
                f"{ct}_var": var_wide[ct],
            })
            data_frames.append(df_pair)
        table = pd.concat(data_frames, axis=1) if data_frames else pd.DataFrame(index=models)

        table["ALL_mean"] = all_mean
        table["ALL_var"]  = all_var
        table.index.name = "model"

        for c in table.columns:
            if c.endswith("_mean"):
                table[c] = table[c].apply(_format_mean)
            elif c.endswith("_var"):
                table[c] = table[c].apply(_format_var)

        fname = f"{level}_{metric}_summary.csv".replace(" ","")
        outp = os.path.join(out_dir, fname)
        table.to_csv(outp)
        print(f"✓ Saved CSV: {outp}")


# ===================================================================
# 10) Main
# ===================================================================
def main():
    df_tidy = load_all_results(MODEL_PATHS, REPI_RANGE)
    model_order = list(MODEL_PATHS.keys())

    # Per-celltype 2×2 violins
    per_ct_dir = os.path.join(OUTPUT_DIR, "per_celltype")
    os.makedirs(per_ct_dir, exist_ok=True)
    celltypes_all = sorted([ct for ct in df_tidy["celltype"].unique() if ct != "ALL"])
    print(f"\n[Info] Found {len(celltypes_all)} celltypes (excluding ALL).")

    for ct in celltypes_all:
        safe_name = str(ct).replace(" ", "_").replace("/", "_").replace("\\", "_")
        out_path = os.path.join(per_ct_dir, f"model_cmp_{safe_name}.png")
        plot_for_celltype(df_tidy, ct, out_path, model_order)

    # Overview (aggregate across all celltypes)
    overview_png = os.path.join(OUTPUT_DIR, "model_cmp_OVERVIEW_all_celltypes.png")
    plot_overview_all_celltypes(df_tidy, overview_png, model_order)

    # Overview (overlay-by-celltype)
    overlay_png = os.path.join(OUTPUT_DIR, "model_cmp_OVERVIEW_overlay-by-celltype.png")
    overlay_violin_by_celltype(df_tidy, overlay_png, model_order, max_celltypes=None)

    # Radar plots (axis-zoomed as requested)
    radar_dir = os.path.join(OUTPUT_DIR, "radar_by_metric_models")
    plot_radar_by_metric_models(df_tidy, radar_dir, model_order, max_celltypes=None, alpha=0.25)

    # ======= REMOVED per your request =======
    # save_all_celltypes_to_pdf(...) and full heatmaps

    # repi=0 heatmaps
    exp00_dir = os.path.join(OUTPUT_DIR, "exp_0_0")
    plot_experiment_00_summary(df_tidy, exp00_dir, model_order)

    # Four CSVs
    csv_dir = os.path.join(OUTPUT_DIR, "metric_csvs")
    save_metric_csvs(df_tidy, csv_dir, model_order)

    # # Global diagnostics
    # diag_dir = os.path.join(OUTPUT_DIR, "diagnostics")
    # plot_diagnostics_scatters(df_tidy, diag_dir, model_order)

    # # New: per-celltype scatter with raw rows & bubble overlap
    # per_ct_scatter_dir = os.path.join(OUTPUT_DIR, "diagnostics_by_celltype")
    # plot_scatter_by_celltype_cells(df_tidy, per_ct_scatter_dir, model_order)

    print("\nDone. All plots & CSVs saved under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()