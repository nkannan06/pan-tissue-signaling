"""
Addition 3: Spatial Communication Mapping.
Simulate the periodic table projected onto a DFU wound cross-section.
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import nnls

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from annotate_atoms import _auto_label, map_atom_to_diseases
from pseudobulk import UNIVERSAL_TYPES
from liana.resource import select_resource
from nmf_decompose import row_normalize

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
FIG_DIR  = "/Users/nitkan/pan-tissue-signaling/figures"
N_TYPES  = len(UNIVERSAL_TYPES)
IDX      = {t: i for i, t in enumerate(UNIVERSAL_TYPES)}
# Macrophage=0, T_cell=1, B_cell=2, NK_cell=3, Dendritic_cell=4, Fibroblast=5
GRID     = 20
SEED     = 42
np.random.seed(SEED)

ATOM_CATEGORY_COLORS = {
    "cancer":       "#d62728",
    "fibrosis":     "#8c564b",
    "inflammation": "#ff7f0e",
    "homeostasis":  "#2196F3",
    "unknown":      "#aaaaaa",
}

ZONE_LABELS = {
    "wound_bed":    {"rows": (8, 12),  "desc": "Wound bed"},
    "wound_edge":   {"rows": (5, 8),   "desc": "Wound edge"},
    "wound_edge2":  {"rows": (12, 15), "desc": "Wound edge"},
    "perilesional": {"rows": (0, 5),   "desc": "Perilesional skin"},
    "perilesional2":{"rows": (15, 20), "desc": "Perilesional skin"},
}


# ── Step 1: Build synthetic spatial grid ─────────────────────────────────────

def assign_cell_composition(row_i, col_j, grid=GRID, wound_type="nonhealing"):
    """
    Return cell type fractions for a spot based on anatomical zone.
    Returns dict {cell_type: fraction}.
    """
    # Wound zones: centre rows 8-12
    dist_from_centre = abs(row_i - grid // 2)
    deep_col = col_j < 5  # deep dermal layer

    if wound_type == "healing":
        # Inflammation receding from centre, homeostatic expanding from edge
        inflam_factor = max(0, 1.0 - dist_from_centre / 8.0) * 0.5  # half of nonhealing
        fibro_factor  = max(0, 1.0 - dist_from_centre / 6.0) * 0.4
        homeost_boost = min(1.0, dist_from_centre / 5.0) * 0.6
    else:
        inflam_factor = max(0, 1.0 - dist_from_centre / 8.0)
        fibro_factor  = max(0, 1.0 - dist_from_centre / 6.0)
        homeost_boost = 0.0

    if dist_from_centre <= 2:    # Wound bed
        comp = {
            "Macrophage":     0.30 + inflam_factor * 0.20,
            "T_cell":         0.15 + inflam_factor * 0.15,
            "B_cell":         0.05,
            "NK_cell":        0.05,
            "Dendritic_cell": 0.05,
            "Fibroblast":     0.35 + fibro_factor  * 0.20,
        }
    elif dist_from_centre <= 5:   # Wound edge
        comp = {
            "Macrophage":     0.20 + inflam_factor * 0.10,
            "T_cell":         0.20 + inflam_factor * 0.10,
            "B_cell":         0.10,
            "NK_cell":        0.10 + homeost_boost * 0.10,
            "Dendritic_cell": 0.10,
            "Fibroblast":     0.25 + fibro_factor  * 0.10,
        }
    else:                         # Perilesional skin
        comp = {
            "Macrophage":     0.08,
            "T_cell":         0.12 + homeost_boost * 0.10,
            "B_cell":         0.12 + homeost_boost * 0.10,
            "NK_cell":        0.20 + homeost_boost * 0.15,
            "Dendritic_cell": 0.15,
            "Fibroblast":     0.25,
        }

    if deep_col:  # Deep dermal: more fibroblast/macrophage
        comp["Fibroblast"]  = min(1.0, comp["Fibroblast"]  + 0.10)
        comp["Macrophage"]  = min(1.0, comp["Macrophage"]  + 0.05)
        comp["NK_cell"]     = max(0.0, comp["NK_cell"]     - 0.05)
        comp["B_cell"]      = max(0.0, comp["B_cell"]      - 0.05)

    # Add noise and renorm
    for ct in comp:
        comp[ct] = max(0, comp[ct] + np.random.normal(0, 0.02))
    total = sum(comp.values())
    return {ct: v/total for ct, v in comp.items()}


def build_spatial_grid(grid=GRID, wound_type="nonhealing"):
    """Returns (grid, grid, N_TYPES) array of cell fractions per spot."""
    spatial = np.zeros((grid, grid, N_TYPES))
    for r in range(grid):
        for c in range(grid):
            comp = assign_cell_composition(r, c, grid, wound_type)
            for ct, frac in comp.items():
                if ct in IDX:
                    spatial[r, c, IDX[ct]] = frac
    return spatial


# ── Step 2: Compute local communication per spot ──────────────────────────────

def compute_spot_flow(spot_comp, neighbor_comps, lr_db, gene_names=None):
    """
    Simplified LR scoring: use cell-type fractions as proxy for expression.
    For each sender(neighbour)-receiver(spot) pair, score = fraction × fraction.
    Returns a flow vector of length N_TYPES×N_TYPES.
    """
    # Aggregate neighbour composition
    if len(neighbor_comps) == 0:
        return np.zeros(N_TYPES * N_TYPES)

    neigh_mean = np.stack(neighbor_comps).mean(axis=0)  # (N_TYPES,)

    # Build N_TYPES × N_TYPES flow matrix: sender=neighbour, receiver=spot
    flow = np.outer(neigh_mean, spot_comp)   # (N_TYPES, N_TYPES) sender→receiver
    return flow.flatten().astype(np.float32)


def compute_all_spots(spatial, lr_db, window=1):
    """
    For each spot, compute flow vector using surrounding neighbourhood.
    Returns flows array (GRID, GRID, N_TYPES²).
    """
    G = spatial.shape[0]
    flows = np.zeros((G, G, N_TYPES * N_TYPES), dtype=np.float32)

    for r in range(G):
        for c in range(G):
            spot_comp = spatial[r, c]                 # (N_TYPES,)
            neighbors = []
            for dr in range(-window, window+1):
                for dc in range(-window, window+1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < G and 0 <= nc < G:
                        neighbors.append(spatial[nr, nc])
            flow = compute_spot_flow(spot_comp, neighbors, lr_db)
            flows[r, c] = flow

    return flows


# ── Step 3: Project onto atom basis ──────────────────────────────────────────

def project_flows_to_atoms(flows, H_real):
    """
    NNLS project each spot's flow vector onto atom basis.
    Returns (GRID, GRID, K) atom weights per spot.
    """
    G = flows.shape[0]
    K = H_real.shape[0]
    W_spatial = np.zeros((G, G, K))

    # Normalize H to same scale as flow vectors
    H_norm = H_real / (np.linalg.norm(H_real, axis=1, keepdims=True) + 1e-12)

    for r in range(G):
        for c in range(G):
            v = flows[r, c]
            v_norm = v / (np.linalg.norm(v) + 1e-12)
            w, _ = nnls(H_norm.T, v_norm)
            s = w.sum()
            W_spatial[r, c] = w / (s + 1e-12) if s > 1e-12 else w

    return W_spatial


# ── Step 4: Visualise ─────────────────────────────────────────────────────────

def spatial_to_rgb(W_spatial, atom_labels):
    """Map dominant atom per spot to category color."""
    G = W_spatial.shape[0]
    dom_atoms = np.argmax(W_spatial, axis=2)    # (G, G)
    color_grid = np.zeros((G, G, 3))

    import matplotlib.colors as mcolors
    for r in range(G):
        for c in range(G):
            k    = dom_atoms[r, c]
            lbl  = atom_labels[k]
            info = map_atom_to_diseases(lbl)
            cat  = info["category"]
            hex_ = ATOM_CATEGORY_COLORS.get(cat, "#aaaaaa")
            color_grid[r, c] = mcolors.to_rgb(hex_)

    return color_grid, dom_atoms


def plot_single_spatial(ax, color_grid, W_spatial, atom_labels, title, grid=GRID):
    ax.imshow(color_grid, origin="upper", aspect="equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Tissue depth (columns)", fontsize=8)
    ax.set_ylabel("Lesion axis (rows)", fontsize=8)

    # Zone annotation lines + labels
    zone_rows = [(5, "Perilesional"), (8, "Wound edge"),
                 (12, "Wound bed"), (15, "Wound edge"), (20, "Perilesional")]
    for r_line, zone_lbl in zone_rows:
        if 0 < r_line < grid:
            ax.axhline(r_line - 0.5, color="white", linewidth=1.0, linestyle="--", alpha=0.7)

    ax.text(grid+0.3, 2.5,  "Perilesional\nskin",   fontsize=6.5, va="center", color="#555")
    ax.text(grid+0.3, 6.5,  "Wound\nedge",          fontsize=6.5, va="center", color="#555")
    ax.text(grid+0.3, 10.0, "Wound\nbed",           fontsize=6.5, va="center", color="#555")
    ax.text(grid+0.3, 13.5, "Wound\nedge",          fontsize=6.5, va="center", color="#555")
    ax.text(grid+0.3, 17.5, "Perilesional\nskin",   fontsize=6.5, va="center", color="#555")

    ax.set_xlim(-0.5, grid + 5)
    ax.set_xticks([]); ax.set_yticks([])


def plot_spatial_map(W_spatial_nh, W_spatial_h, atom_labels, out_path):
    """Side-by-side: non-healing vs healing wound spatial maps."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("white")

    rgb_nh, dom_nh = spatial_to_rgb(W_spatial_nh, atom_labels)
    rgb_h,  dom_h  = spatial_to_rgb(W_spatial_h,  atom_labels)

    plot_single_spatial(axes[0], rgb_nh, W_spatial_nh, atom_labels,
                        "Non-Healing Wound (DFU)\n← locked fibrosis + persistent inflammation")
    plot_single_spatial(axes[1], rgb_h,  W_spatial_h,  atom_labels,
                        "Healing Wound\n← inflammation receding, homeostasis expanding inward")

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=ATOM_CATEGORY_COLORS["fibrosis"],     label="Fibrosis program"),
        mpatches.Patch(color=ATOM_CATEGORY_COLORS["inflammation"],  label="Inflammation program"),
        mpatches.Patch(color=ATOM_CATEGORY_COLORS["cancer"],        label="Tumor/remodeling program"),
        mpatches.Patch(color=ATOM_CATEGORY_COLORS["homeostasis"],   label="Homeostasis program"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Spatial Communication Atlas — DFU Wound Cross-Section\n"
                 "(Each cell colored by dominant NMF atom in 20×20 spatial grid)",
                 fontsize=11, y=1.01)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def print_zonation_stats(W_spatial, atom_labels, wound_type):
    """Print dominant atom category per anatomical zone."""
    G = W_spatial.shape[0]
    zones = {
        "wound_bed":    (8, 12),
        "wound_edge":   (5, 8),
        "perilesional": (0, 5),
    }
    print(f"\n  {wound_type.upper()} — zonation summary:")
    for zone, (r0, r1) in zones.items():
        region = W_spatial[r0:r1]   # (n_rows, G, K)
        dom_k  = int(np.argmax(region.mean(axis=(0, 1))))
        lbl    = atom_labels[dom_k]
        info   = map_atom_to_diseases(lbl)
        w_mean = region.mean(axis=(0, 1))
        top2   = np.argsort(w_mean)[::-1][:2]
        cats   = [map_atom_to_diseases(atom_labels[k])["category"] for k in top2]
        weights= [w_mean[k]*100 for k in top2]
        print(f"    {zone:<15}: dominant={info['category']:<14} "
              f"({cats[0]} {weights[0]:.0f}%, {cats[1]} {weights[1]:.0f}%)")


def run_spatial_mapping():
    H_real      = np.load(f"{DATA_DIR}/H_real.npy")
    K           = H_real.shape[0]
    atom_labels = [_auto_label(H_real[k]) for k in range(K)]

    print("Loading LIANA database...")
    lr_db = select_resource("consensus")

    print(f"\nBuilding spatial grids ({GRID}×{GRID})...")
    spatial_nh = build_spatial_grid(GRID, wound_type="nonhealing")
    spatial_h  = build_spatial_grid(GRID, wound_type="healing")

    print("Computing spot flow vectors...")
    flows_nh = compute_all_spots(spatial_nh, lr_db, window=1)
    flows_h  = compute_all_spots(spatial_h,  lr_db, window=1)

    print("Projecting onto atom basis (NNLS)...")
    W_spatial_nh = project_flows_to_atoms(flows_nh, H_real)
    W_spatial_h  = project_flows_to_atoms(flows_h,  H_real)

    print("\nZonation validation:")
    print_zonation_stats(W_spatial_nh, atom_labels, "non-healing")
    print_zonation_stats(W_spatial_h,  atom_labels, "healing")

    # Save individual spatial map
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    fig.patch.set_facecolor("white")
    rgb_nh, _ = spatial_to_rgb(W_spatial_nh, atom_labels)
    plot_single_spatial(ax, rgb_nh, W_spatial_nh, atom_labels,
                        "DFU Non-Healing Wound — Spatial Atom Map\n(20×20 grid, dominant program per spot)")
    legend_patches = [
        mpatches.Patch(color=ATOM_CATEGORY_COLORS["fibrosis"],    label="Fibrosis"),
        mpatches.Patch(color=ATOM_CATEGORY_COLORS["inflammation"], label="Inflammation"),
        mpatches.Patch(color=ATOM_CATEGORY_COLORS["cancer"],       label="Cancer/remodeling"),
        mpatches.Patch(color=ATOM_CATEGORY_COLORS["homeostasis"],  label="Homeostasis"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path1 = f"{FIG_DIR}/spatial_atom_map.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path1}")

    # Step 5: healer vs non-healer comparison
    path2 = f"{FIG_DIR}/spatial_healer_vs_nonhealer.png"
    plot_spatial_map(W_spatial_nh, W_spatial_h, atom_labels, path2)

    return W_spatial_nh, W_spatial_h, atom_labels


if __name__ == "__main__":
    run_spatial_mapping()
