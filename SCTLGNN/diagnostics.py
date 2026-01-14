# diagnostics.py
import os, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def md5_of_array(a: np.ndarray) -> str:
    a = np.asarray(a)
    return hashlib.md5(a.tobytes()).hexdigest()

def save_json(obj, path):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        # Use the custom encoder via the 'cls' argument
        json.dump(obj, f, indent=2, cls=NpEncoder)

def save_txt(lines, path):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        for ln in lines:
            f.write(str(ln) + "\n")

def save_np(a, path):
    _ensure_dir(os.path.dirname(path))
    np.save(path, a)

def plot_heatmap(df: pd.DataFrame, title: str, out_png: str):
    _ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(6,5))
    plt.imshow(df.values, aspect="auto")
    plt.xticks(range(df.shape[1]), df.columns, rotation=90)
    plt.yticks(range(df.shape[0]), df.index)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_hist(vals, title, out_png):
    _ensure_dir(os.path.dirname(out_png))
    vals = np.asarray(vals)
    plt.figure()
    plt.hist(vals[~np.isnan(vals)], bins=50)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_scatter(y_true, y_pred, title, out_png):
    _ensure_dir(os.path.dirname(out_png))
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = min(20000, y_true.size)  # subsample for speed
    idx = np.random.RandomState(0).choice(y_true.size, n, replace=False)
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[idx], y_pred[idx], s=2, alpha=0.3)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def compare_lists(list_a, list_b):
    A, B = set(list_a), set(list_b)
    inter = A & B
    union = A | B
    return {
        "len_a": len(A),
        "len_b": len(B),
        "len_intersection": len(inter),
        "len_union": len(union),
        "jaccard": (len(inter) / len(union)) if len(union) else 1.0
    }

def save_split_indices(train_idx, valid_idx, out_dir):
    _ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "train_idx.npy"), np.asarray(train_idx, dtype=int))
    np.save(os.path.join(out_dir, "valid_idx.npy"), np.asarray(valid_idx, dtype=int))
    meta = {
        "train_len": int(len(train_idx)),
        "valid_len": int(len(valid_idx)),
        "md5_train": md5_of_array(np.asarray(train_idx, dtype=int)),
        "md5_valid": md5_of_array(np.asarray(valid_idx, dtype=int)),
    }
    save_json(meta, os.path.join(out_dir, "split_meta.json"))

def log_graph_stats(g, out_dir, tag="graph"):
    _ensure_dir(out_dir)
    stats = {}
    try:
        stats["num_nodes_cell"] = int(g.num_nodes("cell"))
        stats["num_edges_cell_cell"] = int(g.num_edges(("cell","cell")))
    except Exception:
        # fallback for homogeneous or other schemas
        stats["num_nodes_total"] = int(sum(g.num_nodes(nt) for nt in g.ntypes)) if hasattr(g, "ntypes") else int(g.num_nodes())
        stats["num_edges_total"] = int(sum(g.num_edges(et) for et in g.etypes)) if hasattr(g, "etypes") else int(g.num_edges())
    save_json(stats, os.path.join(out_dir, f"{tag}_stats.json"))

def save_similarity_matrix(sim_mat_dict, out_dir, tag="similarity"):
    # sim_mat_dict: Dict[str, Dict[str, float]]
    types = sorted(sim_mat_dict.keys())
    df = pd.DataFrame(index=types, columns=types, data=0.0)
    for r in types:
        for c, v in sim_mat_dict[r].items():
            df.loc[r, c] = float(v)
    csv = os.path.join(out_dir, f"{tag}.csv")
    png = os.path.join(out_dir, f"{tag}.png")
    df.to_csv(csv)
    plot_heatmap(df, f"{tag}", png)
    return df, csv, png

def save_markers(marker_map, out_dir, top_n=50):
    # marker_map: Dict[celltype -> List[genes]]
    _ensure_dir(out_dir)
    for ct, genes in marker_map.items():
        save_txt(genes[:top_n], os.path.join(out_dir, f"markers_{ct}.txt"))

def save_weights_per_celltype(weights, obs_celltypes, target_ct, out_dir):
    _ensure_dir(out_dir)
    df = pd.DataFrame({"celltype": list(obs_celltypes), "w": np.asarray(weights, dtype=float)})
    df.to_csv(os.path.join(out_dir, f"weights_{target_ct}.csv"), index=False)
    plot_hist(df["w"].values, f"weights for {target_ct}", os.path.join(out_dir, f"weights_{target_ct}.png"))
    
