"""
Step 10: Pseudo-time ordering + UMAP trajectory map.
Disease progression score = Euclidean distance from healthy centroid in W-space.
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

import umap

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
FIG_DIR  = "/Users/nitkan/pan-tissue-signaling/figures"

SRC_COLORS = {
    "pbmc3k":         "#2196F3",   # blue   — healthy
    "pbmc68k":        "#64B5F6",   # light blue — healthy
    "synthetic_covid":"#FF9800",   # orange — inflammation
    "synthetic_ipf":  "#795548",   # brown  — fibrosis
    "synthetic_tumor":"#F44336",   # red    — cancer
}
SRC_LABELS = {
    "pbmc3k":         "Healthy PBMC (3k)",
    "pbmc68k":        "Healthy PBMC (68k)",
    "synthetic_covid":"COVID-19 (synthetic)",
    "synthetic_ipf":  "IPF fibrotic (synthetic)",
    "synthetic_tumor":"Tumor TME (synthetic)",
}
DISEASE_GROUPS = {
    "healthy":      ["pbmc3k", "pbmc68k"],
    "inflammation": ["synthetic_covid"],
    "fibrosis":     ["synthetic_ipf"],
    "cancer":       ["synthetic_tumor"],
}


def compute_pseudotime(W_norm, meta):
    """
    Progression score = Euclidean distance from healthy centroid in W-space.
    Returns progression_score array (n_samples,) and healthy_centroid (K,).
    """
    healthy_mask = meta["dataset_source"].isin(DISEASE_GROUPS["healthy"]).values
    healthy_centroid = W_norm[healthy_mask].mean(axis=0)
    progression = np.linalg.norm(W_norm - healthy_centroid, axis=1)
    return progression, healthy_centroid


def fit_umap(W_norm, seed=42):
    reducer = umap.UMAP(n_components=2, n_neighbors=8, min_dist=0.3,
                        metric="euclidean", random_state=seed)
    embedding = reducer.fit_transform(W_norm)
    return embedding, reducer


def compute_group_centroids(embedding, meta):
    centroids = {}
    for group, sources in DISEASE_GROUPS.items():
        mask = meta["dataset_source"].isin(sources).values
        if mask.any():
            centroids[group] = embedding[mask].mean(axis=0)
    return centroids


def plot_trajectory_map(embedding, meta, progression, centroids, out_path,
                        extra_points=None, extra_labels=None):
    """
    Scatter: points colored by source.
    Color bar: disease progression score.
    Arrows: healthy centroid → each disease group centroid.
    extra_points: list of (x,y) coords to overlay as prediction markers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: colored by disease group ───────────────────────────────────────
    ax = axes[0]
    for src in meta["dataset_source"].unique():
        mask = (meta["dataset_source"] == src).values
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=SRC_COLORS[src], label=SRC_LABELS[src],
                   s=60, edgecolors="white", linewidths=0.5, zorder=3)

    # Arrows healthy → disease
    arrow_targets = {
        "cancer":       ("#F44336", "Tumor TME"),
        "fibrosis":     ("#795548", "IPF/Fibrosis"),
        "inflammation": ("#FF9800", "COVID/Inflammation"),
    }
    hc = centroids["healthy"]
    for group, (color, label) in arrow_targets.items():
        if group in centroids:
            dc = centroids[group]
            ax.annotate("", xy=dc, xytext=hc,
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=2.5, mutation_scale=18))
            mid = (hc + dc) / 2 + np.array([0.1, 0.1])
            ax.text(mid[0], mid[1], label, fontsize=7.5, color=color,
                    fontweight="bold", ha="center")

    # Mark healthy centroid
    ax.scatter(*hc, marker="*", s=250, c="black", zorder=5, label="Healthy centroid")

    ax.set_title("Cell–Cell Communication Trajectory Map\n(UMAP of atom weight space)",
                 fontsize=10)
    ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
    ax.legend(fontsize=7, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.2)

    # ── Right: colored by progression score ──────────────────────────────────
    ax2 = axes[1]
    norm = Normalize(vmin=progression.min(), vmax=progression.max())
    sc   = ax2.scatter(embedding[:, 0], embedding[:, 1],
                       c=progression, cmap="plasma", s=70,
                       edgecolors="white", linewidths=0.5, norm=norm, zorder=3)
    plt.colorbar(sc, ax=ax2, label="Disease progression score")

    if extra_points is not None:
        for i, (pt, lbl) in enumerate(zip(extra_points, extra_labels or [])):
            ax2.scatter(*pt, marker="D", s=120, c="cyan",
                        edgecolors="black", linewidths=1.2, zorder=6)
            ax2.text(pt[0]+0.05, pt[1]+0.05, lbl, fontsize=7, color="cyan",
                     fontweight="bold")

    ax2.set_title("Disease Progression Score\n(distance from healthy centroid)",
                  fontsize=10)
    ax2.set_xlabel("UMAP-1", fontsize=9); ax2.set_ylabel("UMAP-2", fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return fig


def run_pseudotime():
    W      = np.load(f"{DATA_DIR}/W_real.npy")
    meta   = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    progression, healthy_centroid = compute_pseudotime(W_norm, meta)
    meta["progression_score"] = progression

    print("Progression scores by source:")
    for src in meta["dataset_source"].unique():
        mask = meta["dataset_source"] == src
        vals = progression[mask]
        print(f"  {src:<25} mean={vals.mean():.3f} ± {vals.std():.3f}  "
              f"[{vals.min():.3f}, {vals.max():.3f}]")

    embedding, reducer = fit_umap(W_norm)
    meta["umap1"] = embedding[:, 0]
    meta["umap2"] = embedding[:, 1]
    centroids = compute_group_centroids(embedding, meta)

    print("\nGroup centroids in UMAP space:")
    for g, c in centroids.items():
        print(f"  {g}: ({c[0]:.2f}, {c[1]:.2f})")

    plot_trajectory_map(embedding, meta, progression, centroids,
                        f"{FIG_DIR}/trajectory_map.png")

    # Persist for downstream scripts
    meta.to_csv(f"{DATA_DIR}/meta_with_pseudotime.csv", index=False)
    np.save(f"{DATA_DIR}/W_norm.npy", W_norm)
    np.save(f"{DATA_DIR}/umap_embedding.npy", embedding)
    np.save(f"{DATA_DIR}/healthy_centroid.npy", healthy_centroid)

    print("\nProgression score summary:")
    summary = meta.groupby("dataset_source")["progression_score"].agg(["mean","std"])
    print(summary.sort_values("mean").to_string())
    return meta, embedding, progression, healthy_centroid, reducer


if __name__ == "__main__":
    run_pseudotime()
