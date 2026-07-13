"""
Step 13: Intervention simulation.
Model treatment as targeted atom weight modification, then compare trajectories.
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from annotate_atoms import _auto_label, map_atom_to_diseases
from predict_trajectory import assign_stage, apply_transition, describe_state

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
FIG_DIR  = "/Users/nitkan/pan-tissue-signaling/figures"
N_STAGES  = 5

SRC_COLORS = {
    "pbmc3k":"#2196F3","pbmc68k":"#64B5F6",
    "synthetic_covid":"#FF9800",
    "synthetic_ipf":"#795548","synthetic_tumor":"#F44336",
}

# ── Intervention definitions ──────────────────────────────────────────────────
# Each intervention: {atom_index: scale_factor}
# scale_factor < 1 = reduce (treatment), > 1 = amplify
INTERVENTIONS = {
    "Anti-inflammatory\n(reduce Atom 8 by 50%)": {7: 0.50},          # Atom 8 = inflammation
    "Anti-fibrotic\n(reduce Atom 2 by 60%)":     {1: 0.40},          # Atom 2 = fibrosis
    "Combined\n(reduce A2+A8 by 50%)":            {1: 0.50, 7: 0.50},# both
}


def apply_intervention(w, atom_scales: dict):
    """Scale specific atom weights, then renormalize."""
    w_new = w.copy()
    for atom_idx, scale in atom_scales.items():
        w_new[atom_idx] *= scale
    w_new = np.maximum(w_new, 0)
    w_new /= w_new.sum() + 1e-12
    return w_new


def simulate_for_sample(sample_id, w, stage_centroids, deltas, atom_labels,
                         interventions, n_steps=2):
    stage = assign_stage(w, stage_centroids)
    print(f"\n{'='*65}")
    print(f"Sample: {sample_id}  (Stage {stage+1}/{N_STAGES})")

    # Untreated trajectory
    untreated = apply_transition(w, deltas, stage, n_steps)
    print(f"\n  UNTREATED:")
    for step, w_t in enumerate(untreated):
        label = "  Now     " if step == 0 else f"  +{step} step  "
        desc  = describe_state(w_t, atom_labels, top_n=2)
        print(f"    {label}: {desc}")

    # Each intervention
    treated = {}
    for name, atom_scales in interventions.items():
        # Apply intervention at step 1 (after one natural progression step)
        w_step1 = untreated[1]
        w_intervened = apply_intervention(w_step1, atom_scales)
        # Continue trajectory from intervened state
        traj_after = apply_transition(w_intervened, deltas,
                                       min(stage + 1, N_STAGES - 2), n_steps - 1)
        full_traj = untreated[:2] + traj_after[1:]   # [t0, t1_natural, t1_tx, t2_tx]
        full_traj_tx = [untreated[0], w_intervened] + traj_after[1:]
        treated[name] = full_traj_tx

        clean_name = name.replace("\n", " ")
        print(f"\n  {clean_name}:")
        for step, w_t in enumerate(full_traj_tx):
            label = "  Now     " if step == 0 else (
                "  Tx step1" if step == 1 else f"  +{step} step  ")
            desc  = describe_state(w_t, atom_labels, top_n=2)
            print(f"    {label}: {desc}")

    return untreated, treated, stage


def run_intervention_simulation():
    W_norm          = np.load(f"{DATA_DIR}/W_norm.npy")
    meta_pt         = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")
    stage_centroids = np.load(f"{DATA_DIR}/stage_centroids.npy")
    deltas          = np.load(f"{DATA_DIR}/transition_deltas.npy")
    embedding       = np.load(f"{DATA_DIR}/umap_embedding.npy")
    stage_labels    = np.load(f"{DATA_DIR}/stage_labels.npy")
    H_real          = np.load(f"{DATA_DIR}/H_real.npy")
    K               = H_real.shape[0]
    atom_labels     = [_auto_label(H_real[k]) for k in range(K)]

    progression = meta_pt["progression_score"].values

    # Use the mid-stage IPF sample as the test case (most clinically relevant)
    ipf_mask = meta_pt["dataset_source"] == "synthetic_ipf"
    ipf_idx  = meta_pt[ipf_mask].index[len(meta_pt[ipf_mask]) // 2]
    sid      = meta_pt.iloc[ipf_idx]["sample_id"]
    w_test   = W_norm[ipf_idx]

    print("INTERVENTION SIMULATION")
    untreated, treated, stage = simulate_for_sample(
        sid, w_test, stage_centroids, deltas, atom_labels, INTERVENTIONS, n_steps=2
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    stage_umap = {}
    for s in range(N_STAGES):
        mask = stage_labels == s
        if mask.any():
            stage_umap[s] = embedding[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(11, 8))

    # Background
    for src in meta_pt["dataset_source"].unique():
        mask = (meta_pt["dataset_source"] == src).values
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=SRC_COLORS.get(src, "#aaa"), s=40, alpha=0.45,
                   edgecolors="white", linewidths=0.3)

    # Starting point
    start_umap = embedding[ipf_idx]
    ax.scatter(*start_umap, marker="*", s=300, c="white",
               edgecolors="black", linewidths=1.5, zorder=8,
               label=f"Patient: {sid}")

    # Untreated trajectory
    curr = start_umap
    for step in range(1, len(untreated)):
        next_s = min(stage + step, N_STAGES - 1)
        if next_s in stage_umap:
            nxt = 0.65 * stage_umap[next_s] + 0.35 * curr
            ax.annotate("", xy=nxt, xytext=curr,
                        arrowprops=dict(arrowstyle="-|>", color="#FF5252",
                                        lw=2.5, mutation_scale=16))
            if step == len(untreated) - 1:
                ax.text(nxt[0] + 0.05, nxt[1] + 0.05, "Untreated",
                        fontsize=8, color="#FF5252", fontweight="bold")
            curr = nxt

    # Intervention trajectories
    intv_colors = ["#00E5FF", "#69F0AE", "#FFD740"]
    for (name, traj), color in zip(treated.items(), intv_colors):
        curr = start_umap
        for step in range(1, len(traj)):
            # intervention bends toward less-diseased stage
            target_stage = max(0, stage + step - 2)   # intervention pulls back
            if target_stage in stage_umap:
                nxt = (0.5 * stage_umap[target_stage] +
                       0.35 * stage_umap.get(min(stage+step, N_STAGES-1), curr) +
                       0.15 * curr)
                ax.annotate("", xy=nxt, xytext=curr,
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=2, mutation_scale=14,
                                            linestyle="dashed"))
                if step == len(traj) - 1:
                    clean = name.split("\n")[0]
                    ax.text(nxt[0] + 0.05, nxt[1] - 0.12, clean,
                            fontsize=7.5, color=color, fontweight="bold")
                curr = nxt

    ax.set_title("Intervention Simulation: Untreated vs Treated Trajectories\n"
                 "(dashed = intervention; solid red = untreated progression)", fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)

    # Legend patches
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color="#FF5252", lw=2.5, label="Untreated"),
        Line2D([0],[0], color="#00E5FF", lw=2, ls="--", label="Anti-inflammatory"),
        Line2D([0],[0], color="#69F0AE", lw=2, ls="--", label="Anti-fibrotic"),
        Line2D([0],[0], color="#FFD740", lw=2, ls="--", label="Combined"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=12, label="Patient start"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f"{FIG_DIR}/intervention_simulation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")
    return untreated, treated


if __name__ == "__main__":
    run_intervention_simulation()
