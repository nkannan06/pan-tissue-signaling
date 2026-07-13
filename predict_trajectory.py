"""
Step 12: Trajectory predictor.
Given atom weights w, find current stage, apply transition deltas,
predict 1-step and 2-step ahead.
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from annotate_atoms import _auto_label, map_atom_to_diseases

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
FIG_DIR  = "/Users/nitkan/pan-tissue-signaling/figures"
N_STAGES  = 5

SRC_COLORS = {
    "pbmc3k":"#2196F3", "pbmc68k":"#64B5F6",
    "synthetic_covid":"#FF9800",
    "synthetic_ipf":"#795548", "synthetic_tumor":"#F44336",
}


def assign_stage(w, stage_centroids):
    """Find closest stage centroid in atom-weight space."""
    dists = [np.linalg.norm(w - stage_centroids[s]) for s in range(len(stage_centroids))]
    return int(np.argmin(dists))


def apply_transition(w, deltas, stage, n_steps=1):
    """
    Apply n_steps of transition deltas starting from stage.
    Returns list of w vectors: [w_step1, w_step2, ...].
    """
    trajectory = [w.copy()]
    curr = w.copy()
    curr_stage = stage
    for _ in range(n_steps):
        if curr_stage < len(deltas):
            curr = curr + deltas[curr_stage]
            curr = np.maximum(curr, 0)           # non-negative
            curr = curr / (curr.sum() + 1e-12)   # renormalize
            curr_stage = min(curr_stage + 1, len(deltas) - 1)
        trajectory.append(curr.copy())
    return trajectory


def describe_state(w, atom_labels, top_n=3):
    """Human-readable description of top atom contributions."""
    order = np.argsort(w)[::-1][:top_n]
    parts = []
    for k in order:
        lbl  = atom_labels[k][:25]
        info = map_atom_to_diseases(atom_labels[k])
        cat  = info["category"]
        parts.append(f"{w[k]*100:.0f}% {cat} (A{k+1}:{lbl})")
    return ", ".join(parts)


def predict_for_sample(sample_id, w, stage_centroids, deltas, atom_labels, n_steps=2):
    stage = assign_stage(w, stage_centroids)
    trajectory = apply_transition(w, deltas, stage, n_steps=n_steps)

    print(f"\n{'─'*65}")
    print(f"Sample: {sample_id}  (Stage {stage+1}/{N_STAGES})")
    for step, w_t in enumerate(trajectory):
        label = "Current state" if step == 0 else f"Predicted step +{step}"
        desc  = describe_state(w_t, atom_labels)
        print(f"  {label:>20}: {desc}")

    return trajectory, stage


def test_trajectory_predictions():
    W_norm         = np.load(f"{DATA_DIR}/W_norm.npy")
    meta_pt        = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")
    stage_centroids= np.load(f"{DATA_DIR}/stage_centroids.npy")
    deltas         = np.load(f"{DATA_DIR}/transition_deltas.npy")
    embedding      = np.load(f"{DATA_DIR}/umap_embedding.npy")
    H_real         = np.load(f"{DATA_DIR}/H_real.npy")
    K              = H_real.shape[0]
    atom_labels    = [_auto_label(H_real[k]) for k in range(K)]

    progression = meta_pt["progression_score"].values

    # Pick 3 test samples by progression stage
    idx_early = int(np.argmin(progression))                             # most healthy
    idx_mid   = int(np.argsort(progression)[len(progression)//2])      # median
    idx_late  = int(np.argmax(progression))                             # most diseased
    test_cases = [
        (idx_early, "early-stage (closest to healthy)"),
        (idx_mid,   "mid-stage"),
        (idx_late,  "late-stage (furthest from healthy)"),
    ]

    print(f"{'='*65}")
    print("TRAJECTORY PREDICTIONS")
    print(f"{'='*65}")

    all_trajectories = []
    for idx, desc in test_cases:
        sid = meta_pt.iloc[idx]["sample_id"]
        src = meta_pt.iloc[idx]["dataset_source"]
        w   = W_norm[idx]
        print(f"\n[{desc}] — {sid} ({src})")
        traj, stage = predict_for_sample(sid, w, stage_centroids, deltas,
                                          atom_labels, n_steps=2)
        all_trajectories.append({
            "idx": idx, "sid": sid, "src": src, "desc": desc,
            "stage": stage, "trajectory": traj,
            "umap": embedding[idx],
        })

    # ── Plot predictions on UMAP ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background scatter
    for src in meta_pt["dataset_source"].unique():
        mask = (meta_pt["dataset_source"] == src).values
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=SRC_COLORS.get(src, "#aaa"), s=45, alpha=0.55,
                   edgecolors="white", linewidths=0.4)

    # Trajectory arrows for each test sample
    traj_colors = ["#00E5FF", "#76FF03", "#FF4081"]
    traj_markers= ["o", "s", "^"]

    # We project each predicted w vector onto UMAP via nearest neighbor
    # (UMAP reducer not saved — use stage_centroids as proxy positions)
    # Map stage centroid positions to UMAP by finding nearest sample
    stage_umap = {}
    stage_labels_arr = np.load(f"{DATA_DIR}/stage_labels.npy")
    for s in range(N_STAGES):
        mask = stage_labels_arr == s
        if mask.any():
            stage_umap[s] = embedding[mask].mean(axis=0)

    for tc, color, marker in zip(all_trajectories, traj_colors, traj_markers):
        umap_pt = tc["umap"]
        stage   = tc["stage"]
        ax.scatter(*umap_pt, c=color, marker=marker, s=180,
                   edgecolors="black", linewidths=1.5, zorder=7,
                   label=f"{tc['desc']}")

        # Draw arrows: current → step+1 → step+2 via stage positions
        curr_pos = umap_pt
        for step in range(1, len(tc["trajectory"])):
            next_stage = min(stage + step, N_STAGES - 1)
            if next_stage in stage_umap:
                # interpolate toward next stage centroid
                next_pos = 0.6 * stage_umap[next_stage] + 0.4 * curr_pos
                ax.annotate("", xy=next_pos, xytext=curr_pos,
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=2, mutation_scale=14, alpha=0.85))
                curr_pos = next_pos

    ax.set_title("Trajectory Predictions in Atom-Weight UMAP Space\n"
                 "(arrows show predicted progression direction)", fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # Source legend
    from matplotlib.lines import Line2D
    src_handles = [Line2D([0],[0], marker="o", color="w",
                          markerfacecolor=SRC_COLORS[s], markersize=8,
                          label=s.replace("synthetic_","").replace("pbmc","PBMC "))
                   for s in SRC_COLORS]
    ax.legend(handles=src_handles, fontsize=7, loc="upper left", framealpha=0.85)

    plt.tight_layout()
    path = f"{FIG_DIR}/trajectory_predictions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")
    return all_trajectories


if __name__ == "__main__":
    test_trajectory_predictions()
