"""
Step 6: Visualize NMF atoms as heatmaps + W matrix sample loadings.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import product

UNIVERSAL_TYPES = ["Macrophage", "T_cell", "B_cell", "NK_cell", "Dendritic_cell", "Fibroblast"]
N_TYPES = len(UNIVERSAL_TYPES)
SHORT = ["Mac", "Tcell", "Bcell", "NK", "DC", "Fibro"]
OUT_DIR = "figures"


def auto_label(atom_vec: np.ndarray, top_n: int = 2) -> str:
    """Return top_n sender→receiver edges by weight as a string label."""
    mat = atom_vec.reshape(N_TYPES, N_TYPES)
    pairs = []
    for i, j in product(range(N_TYPES), repeat=2):
        pairs.append((mat[i, j], UNIVERSAL_TYPES[i], UNIVERSAL_TYPES[j]))
    pairs.sort(reverse=True)
    parts = [f"{s}→{r}" for _, s, r in pairs[:top_n]]
    return " / ".join(parts)


def plot_atoms(H: np.ndarray, out_dir: str = OUT_DIR):
    """Plot each NMF atom (row of H) as a 6×6 heatmap."""
    import os; os.makedirs(out_dir, exist_ok=True)
    K = H.shape[0]
    ncols = min(K, 4)
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for k in range(K):
        mat = H[k].reshape(N_TYPES, N_TYPES)
        label = auto_label(H[k])
        ax = axes[k]
        sns.heatmap(
            mat,
            ax=ax,
            xticklabels=SHORT,
            yticklabels=SHORT,
            cmap="YlOrRd",
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            cbar_kws={"shrink": 0.7},
            vmin=0,
        )
        ax.set_title(f"Atom {k+1}:\n{label}", fontsize=9, pad=6)
        ax.set_xlabel("Receiver", fontsize=8)
        ax.set_ylabel("Sender", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for k in range(K, len(axes)):
        axes[k].set_visible(False)

    plt.suptitle(f"NMF Atoms (K={K}) — Pan-Tissue Signaling Periodic Table", fontsize=12, y=1.01)
    plt.tight_layout()
    path = f"{out_dir}/atoms_K{K}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_W_heatmap(W: np.ndarray, labels: list, group_names: list, out_dir: str = OUT_DIR):
    """Plot sample loadings (W matrix) heatmap, grouped by ground truth."""
    import os; os.makedirs(out_dir, exist_ok=True)
    K = W.shape[1]

    # Normalize rows so loadings are comparable
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    df = pd.DataFrame(W_norm, index=labels, columns=[f"Atom {k+1}" for k in range(K)])

    # Add group annotation for color bar
    group_of = [lbl.rsplit("_", 1)[0] for lbl in labels]
    palette = dict(zip(group_names, sns.color_palette("Set2", len(group_names))))
    row_colors = pd.Series(group_of, index=labels).map(palette)

    fig, ax = plt.subplots(figsize=(max(8, K * 1.5), 12))
    sns.heatmap(
        df,
        ax=ax,
        cmap="Blues",
        linewidths=0.2,
        cbar_kws={"label": "Normalized loading", "shrink": 0.5},
        yticklabels=True,
    )
    ax.set_title(f"Sample Loadings on NMF Atoms (K={K})", fontsize=12)
    ax.set_xlabel("Atom")
    ax.set_ylabel("Sample")
    ax.tick_params(axis="y", labelsize=6)

    # Add colored group labels on the left
    n_per_group = len(labels) // len(group_names)
    for i, g in enumerate(group_names):
        mid = i * n_per_group + n_per_group / 2
        ax.text(
            -0.5, mid, g.replace("_", " "),
            va="center", ha="right", fontsize=8,
            color=palette[g], fontweight="bold",
            transform=ax.get_yaxis_transform(),
        )

    plt.tight_layout()
    path = f"{out_dir}/W_loadings_K{K}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def print_summary(H: np.ndarray, W: np.ndarray, labels: list, group_names: list,
                  error: float, stability: float):
    K = H.shape[0]
    n_per_group = len(labels) // len(group_names)

    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY  K={K}")
    print(f"Reconstruction error : {error:.4f}")
    print(f"Stability score      : {stability:.4f}")
    print("=" * 60)

    # For each atom, find which group loads most heavily on it
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    for k in range(K):
        atom_label = auto_label(H[k])
        # mean loading per group
        group_loads = {}
        for gi, g in enumerate(group_names):
            sl = slice(gi * n_per_group, (gi + 1) * n_per_group)
            group_loads[g] = W_norm[sl, k].mean()
        top_group = max(group_loads, key=group_loads.get)
        top_val = group_loads[top_group]
        print(f"\nAtom {k+1}: {atom_label}")
        print(f"  Top group → {top_group} (mean loading={top_val:.3f})")
        for g, v in sorted(group_loads.items(), key=lambda x: -x[1]):
            bar = "█" * int(v * 30)
            print(f"    {g:<25} {bar} {v:.3f}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, ".")
    from generate_synthetic import build_master_matrix

    # Load saved results (run nmf_decompose.py first)
    W = np.load("data/W.npy")
    H = np.load("data/H.npy")
    labels = pd.read_csv("data/labels_synthetic.csv", header=None)[0].tolist()
    group_names = ["tumor_TME", "chronic_inflammation", "healthy", "fibrotic"]

    # We need error/stability — recompute quickly
    from nmf_decompose import row_normalize, run_nmf, stability_score
    M = np.load("data/M_synthetic.npy")
    M_norm = row_normalize(M)
    K = H.shape[0]
    _, _, error = run_nmf(M_norm, K)
    stability = stability_score(M_norm, K, n_runs=10)

    os.makedirs("figures", exist_ok=True)
    plot_atoms(H, out_dir="figures")
    plot_W_heatmap(W, labels, group_names, out_dir="figures")
    print_summary(H, W, labels, group_names, error, stability)
