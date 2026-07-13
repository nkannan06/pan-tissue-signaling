"""
Visualize real-data NMF atoms, W heatmap, and compare to synthetic atoms.
"""

import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

sys.path.insert(0, ".")
from visualize_atoms import plot_atoms, auto_label, UNIVERSAL_TYPES, SHORT, OUT_DIR

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
FIG_DIR  = "/Users/nitkan/pan-tissue-signaling/figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ── 1. Load results ───────────────────────────────────────────────────────────

W_real   = np.load(f"{DATA_DIR}/W_real.npy")
H_real   = np.load(f"{DATA_DIR}/H_real.npy")
H_synth  = np.load(f"{DATA_DIR}/H.npy")          # synthetic atoms K=4
meta     = pd.read_csv(f"{DATA_DIR}/metadata.csv")
labels   = meta["sample_id"].tolist()
sources  = meta["dataset_source"].tolist()

K_real  = H_real.shape[0]
K_synth = H_synth.shape[0]
print(f"Real K={K_real}, Synthetic K={K_synth}")


# ── 2. Atom heatmaps ─────────────────────────────────────────────────────────

plot_atoms(H_real, out_dir=FIG_DIR)   # saves figures/atoms_K8.png


# ── 3. W heatmap colored by dataset source ───────────────────────────────────

def plot_W_real(W, labels, sources, out_dir):
    K = W.shape[1]
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    df = pd.DataFrame(W_norm, index=labels,
                      columns=[f"Atom {k+1}" for k in range(K)])

    unique_src = list(dict.fromkeys(sources))
    palette = dict(zip(unique_src, sns.color_palette("Set2", len(unique_src))))
    row_colors = pd.Series(sources, index=labels).map(palette)

    fig, (ax_heat, ax_leg) = plt.subplots(
        1, 2, figsize=(K * 1.4 + 2, 14),
        gridspec_kw={"width_ratios": [K * 1.4, 1.5]}
    )
    sns.heatmap(df, ax=ax_heat, cmap="Blues", linewidths=0.15,
                cbar_kws={"label": "Norm. loading", "shrink": 0.4},
                yticklabels=True)
    ax_heat.set_title(f"Sample Loadings — Real Data (K={K})", fontsize=11)
    ax_heat.tick_params(axis="y", labelsize=6)

    # Colored group separators
    current_src, start_row = None, 0
    for i, src in enumerate(sources + [None]):
        if src != current_src and current_src is not None:
            mid = (start_row + i) / 2
            ax_heat.axhline(i, color="white", lw=1.5)
            ax_heat.text(-0.3, mid, current_src.replace("_", "\n"),
                         ha="right", va="center", fontsize=7,
                         color=palette[current_src], fontweight="bold",
                         transform=ax_heat.get_yaxis_transform())
            start_row = i
        current_src = src

    # Legend
    ax_leg.axis("off")
    for i, (src, color) in enumerate(palette.items()):
        ax_leg.add_patch(plt.Rectangle((0, 1 - (i + 1) * 0.12), 0.3, 0.09,
                                        color=color, transform=ax_leg.transAxes))
        ax_leg.text(0.38, 1 - (i + 0.5) * 0.12, src.replace("_", " "),
                    va="center", fontsize=8, transform=ax_leg.transAxes)

    plt.tight_layout()
    path = f"{out_dir}/W_real_loadings_K{K}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


plot_W_real(W_real, labels, sources, FIG_DIR)


# ── 4. Cosine similarity: real atoms vs synthetic atoms ──────────────────────

def cosine_sim_matrix(A, B):
    """A: (Ka, D), B: (Kb, D) → (Ka, Kb) cosine sim."""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


sim = cosine_sim_matrix(H_real, H_synth)   # (8, 4)

print("\n--- Cosine similarity: Real atoms (rows) vs Synthetic atoms (cols) ---")
synth_labels = [auto_label(H_synth[k]) for k in range(K_synth)]
real_labels  = [auto_label(H_real[k])  for k in range(K_real)]
sim_df = pd.DataFrame(sim, index=[f"R{k+1}" for k in range(K_real)],
                      columns=[f"S{k+1}" for k in range(K_synth)])
print(sim_df.round(3).to_string())

# Best-match for each real atom
print("\n--- Best synthetic match per real atom ---")
for k in range(K_real):
    best_s = int(np.argmax(sim[k]))
    print(f"  Real Atom {k+1}: {real_labels[k]}")
    print(f"    → Synthetic Atom {best_s+1}: {synth_labels[best_s]}  (cosine={sim[k,best_s]:.3f})")


# ── 5. Plot cosine similarity heatmap ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 7))
sns.heatmap(sim_df, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
            annot=True, fmt=".2f", linewidths=0.5,
            xticklabels=[f"S{i+1}\n{l[:30]}" for i, l in enumerate(synth_labels)],
            yticklabels=[f"R{i+1}: {l[:28]}" for i, l in enumerate(real_labels)])
ax.set_title("Real vs Synthetic Atom Cosine Similarity", fontsize=10)
ax.set_xlabel("Synthetic atoms")
ax.set_ylabel("Real atoms")
plt.tight_layout()
path = f"{FIG_DIR}/atom_cosine_similarity.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {path}")


# ── 6. Per-dataset atom loading summary ──────────────────────────────────────

W_norm = W_real / (W_real.sum(axis=1, keepdims=True) + 1e-12)
meta["dominant_atom"] = np.argmax(W_norm, axis=1) + 1

print("\n--- Dominant atom per dataset source ---")
for src in meta["dataset_source"].unique():
    mask = meta["dataset_source"] == src
    dom = meta.loc[mask, "dominant_atom"].value_counts()
    top_atom = dom.index[0]
    top_label = real_labels[top_atom - 1]
    print(f"  {src:<22} → Atom {top_atom}: {top_label}")


# ── 7. Final printed summary ─────────────────────────────────────────────────

from nmf_decompose import row_normalize, run_nmf, stability_score
M_real_arr = np.load(f"{DATA_DIR}/M_real.npy")
M_norm_real = row_normalize(M_real_arr)
_, _, err_real = run_nmf(M_norm_real, K_real)
stab_real = stability_score(M_norm_real, K_real, n_runs=5)

print(f"\n{'='*65}")
print(f"FINAL SUMMARY — Real Data NMF")
print(f"{'='*65}")
print(f"  M_real shape    : {M_real_arr.shape}")
print(f"  Selected K      : {K_real}")
print(f"  Recon error     : {err_real:.4f}")
print(f"  Stability score : {stab_real:.4f}")
print()
print("  Validation — do real atoms match synthetic ground truth?")
fibrosis_check = any(sim[k, 1] > 0.7 for k in range(K_real))   # synth Atom2 = fibrotic
inflam_check   = any(sim[k, 2] > 0.7 for k in range(K_real))   # synth Atom3 = inflammation
homeo_check    = any(sim[k, 3] > 0.7 for k in range(K_real))   # synth Atom4 = healthy
print(f"  Fibrosis atom (Fibroblast↔Fibroblast)  found: {'✓' if fibrosis_check else '✗'} "
      f"(max cos={sim[:,1].max():.3f})")
print(f"  Inflammation atom (T_cell↔Macrophage)  found: {'✓' if inflam_check else '✗'} "
      f"(max cos={sim[:,2].max():.3f})")
print(f"  Homeostatic atom (diffuse low-signal)  found: {'✓' if homeo_check else '✗'} "
      f"(max cos={sim[:,3].max():.3f})")

concept_valid = fibrosis_check or inflam_check or homeo_check
print(f"\n  Periodic table concept validated: {'YES' if concept_valid else 'PARTIAL'}")
