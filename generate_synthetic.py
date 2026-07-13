"""
Generate synthetic M matrix (n_samples × 30) with realistic group structure.
4 tissue-disease groups × 10 samples each = 40 samples.
Each row is a flattened 6×6 sender-receiver flow matrix.
"""

import numpy as np
import pandas as pd

UNIVERSAL_TYPES = ["Macrophage", "T_cell", "B_cell", "NK_cell", "Dendritic_cell", "Fibroblast"]
N_TYPES = len(UNIVERSAL_TYPES)
IDX = {t: i for i, t in enumerate(UNIVERSAL_TYPES)}


def edge_idx(sender: str, receiver: str) -> int:
    return IDX[sender] * N_TYPES + IDX[receiver]


def base_matrix() -> np.ndarray:
    """Start from a low uniform baseline."""
    return np.full((N_TYPES, N_TYPES), 0.1, dtype=np.float64)


def set_edge(mat, sender, receiver, value):
    mat[IDX[sender], IDX[receiver]] = value


def generate_group(template_fn, n_samples=10, noise_frac=0.1, rng=None) -> np.ndarray:
    """
    Generate n_samples rows, each a noisy version of the template 6×6 matrix.
    Returns (n_samples, 36) array (flattened).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    rows = []
    for _ in range(n_samples):
        mat = template_fn()
        noise = rng.normal(0, noise_frac * mat.mean(), size=mat.shape)
        mat = np.maximum(mat + noise, 0)
        rows.append(mat.flatten())
    return np.array(rows)


# ── Group templates ───────────────────────────────────────────────────────────

def tumor_tme():
    """High Macrophage→T_cell, Fibroblast→Macrophage; low B→NK."""
    m = base_matrix()
    set_edge(m, "Macrophage",   "T_cell",    0.9)
    set_edge(m, "Macrophage",   "NK_cell",   0.7)
    set_edge(m, "Fibroblast",   "Macrophage",0.85)
    set_edge(m, "Fibroblast",   "T_cell",    0.6)
    set_edge(m, "Dendritic_cell","T_cell",   0.5)
    set_edge(m, "B_cell",       "NK_cell",   0.05)  # suppressed
    return m


def chronic_inflammation():
    """High T_cell→Macrophage, Macrophage→Fibroblast; elevated immune edges."""
    m = base_matrix() * 1.5   # elevated baseline
    set_edge(m, "T_cell",      "Macrophage",  0.9)
    set_edge(m, "T_cell",      "Dendritic_cell", 0.75)
    set_edge(m, "Macrophage",  "Fibroblast",  0.85)
    set_edge(m, "Macrophage",  "T_cell",      0.7)
    set_edge(m, "NK_cell",     "Macrophage",  0.6)
    set_edge(m, "B_cell",      "T_cell",      0.55)
    return m


def healthy_tissue():
    """Moderate, even communication; low variance."""
    m = np.full((N_TYPES, N_TYPES), 0.35, dtype=np.float64)
    # Slight immune surveillance edges
    set_edge(m, "NK_cell",  "Macrophage", 0.45)
    set_edge(m, "T_cell",   "B_cell",     0.4)
    return m


def fibrotic():
    """High Fibroblast autocrine + Fibroblast→Macrophage; low T_cell edges."""
    m = base_matrix()
    set_edge(m, "Fibroblast",  "Fibroblast",  0.95)  # autocrine
    set_edge(m, "Fibroblast",  "Macrophage",  0.8)
    set_edge(m, "Macrophage",  "Fibroblast",  0.75)
    set_edge(m, "T_cell",      "Macrophage",  0.08)  # suppressed
    set_edge(m, "T_cell",      "B_cell",      0.06)  # suppressed
    return m


# ── Build master matrix ───────────────────────────────────────────────────────

def build_master_matrix(n_per_group=10, noise_frac=0.1, seed=42) -> tuple:
    """
    Returns
    -------
    M      : (40, 36) float64
    labels : list of str, length 40
    groups : dict {group_name: slice}
    """
    rng = np.random.default_rng(seed)

    groups = {
        "tumor_TME":           tumor_tme,
        "chronic_inflammation": chronic_inflammation,
        "healthy":             healthy_tissue,
        "fibrotic":            fibrotic,
    }

    blocks = []
    labels = []
    for gname, fn in groups.items():
        block = generate_group(fn, n_samples=n_per_group, noise_frac=noise_frac, rng=rng)
        blocks.append(block)
        labels += [f"{gname}_{i+1:02d}" for i in range(n_per_group)]

    M = np.vstack(blocks)
    return M, labels, list(groups.keys())


if __name__ == "__main__":
    M, labels, group_names = build_master_matrix()
    print(f"M shape: {M.shape}")
    print(f"Labels: {labels[:3]} ... {labels[-3:]}")
    print(f"Groups: {group_names}")
    print(f"M min={M.min():.4f}, max={M.max():.4f}, mean={M.mean():.4f}")

    # Verify group signals are distinct
    print("\nPer-group mean flow magnitude:")
    for i, g in enumerate(group_names):
        block = M[i*10:(i+1)*10]
        print(f"  {g:<25} mean={block.mean():.3f}  std={block.std():.3f}")

    # Save for downstream use
    np.save("data/M_synthetic.npy", M)
    pd.Series(labels).to_csv("data/labels_synthetic.csv", index=False, header=False)
    print("\nSaved: data/M_synthetic.npy, data/labels_synthetic.csv")
