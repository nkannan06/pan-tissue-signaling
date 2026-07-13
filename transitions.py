"""
Step 11: Transition probability matrix across disease progression stages.
Bins samples into 5 quintile stages and computes atom change signatures.
"""

import sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from annotate_atoms import _auto_label, map_atom_to_diseases

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
N_STAGES  = 5


def build_transition_table(W_norm, meta_pt):
    """
    Bin samples into N_STAGES quintiles by progression score.
    For each adjacent stage pair, compute mean atom weight delta.
    Returns transition_df and stage_centroids.
    """
    scores    = meta_pt["progression_score"].values
    quantiles = np.quantile(scores, np.linspace(0, 1, N_STAGES + 1))
    stage_labels = np.digitize(scores, quantiles[1:-1])   # 0-indexed stages 0..4

    meta_pt = meta_pt.copy()
    meta_pt["stage"] = stage_labels
    K = W_norm.shape[1]

    # Mean W per stage
    stage_centroids = {}
    for s in range(N_STAGES):
        mask = stage_labels == s
        if mask.sum() > 0:
            stage_centroids[s] = W_norm[mask].mean(axis=0)
        else:
            stage_centroids[s] = np.zeros(K)

    # Atom labels
    H_real       = np.load(f"{DATA_DIR}/H_real.npy")
    atom_labels  = [_auto_label(H_real[k]) for k in range(K)]
    short_labels = [f"A{k+1}:{lbl[:20]}" for k, lbl in enumerate(atom_labels)]

    rows = []
    for s in range(N_STAGES - 1):
        c_curr = stage_centroids[s]
        c_next = stage_centroids[s + 1]
        delta  = c_next - c_curr

        increasing = [(k, delta[k]) for k in range(K) if delta[k] > 0.005]
        decreasing = [(k, delta[k]) for k in range(K) if delta[k] < -0.005]
        increasing.sort(key=lambda x: -x[1])
        decreasing.sort(key=lambda x: x[1])

        # Samples in each stage
        n_curr = (stage_labels == s).sum()
        n_next = (stage_labels == s+1).sum()
        prog_range_curr = scores[stage_labels == s]
        prog_range_next = scores[stage_labels == s+1]

        rows.append({
            "stage_transition": f"Stage {s+1}→{s+2}",
            "n_samples_from":   n_curr,
            "n_samples_to":     n_next,
            "prog_score_range": f"{prog_range_curr.min():.3f}–{prog_range_next.max():.3f}",
            "atoms_increasing": ", ".join(f"A{k+1}(+{d:.3f})" for k, d in increasing[:3]),
            "atoms_decreasing": ", ".join(f"A{k+1}({d:.3f})" for k, d in decreasing[:3]),
            "dominant_change":  f"A{increasing[0][0]+1}↑" if increasing else "stable",
            "delta": delta,
        })

    trans_df = pd.DataFrame(rows)

    # Print table
    print(f"\n{'='*75}")
    print("TRANSITION TABLE (disease progression stages)")
    print(f"{'='*75}")
    print(f"{'Transition':<14} {'n→n':<7} {'Score range':<18} "
          f"{'Atoms ↑':<28} {'Atoms ↓':<28} {'Dominant'}")
    print("-" * 105)
    for _, row in trans_df.iterrows():
        print(f"{row['stage_transition']:<14} "
              f"{row['n_samples_from']}→{row['n_samples_to']:<5} "
              f"{row['prog_score_range']:<18} "
              f"{row['atoms_increasing']:<28} "
              f"{row['atoms_decreasing']:<28} "
              f"{row['dominant_change']}")

    print(f"\nAtom key:")
    for k, lbl in enumerate(atom_labels):
        print(f"  A{k+1}: {lbl}")

    # Print stage compositions
    print(f"\n{'─'*60}")
    print("Stage composition (dominant source per stage):")
    for s in range(N_STAGES):
        mask = stage_labels == s
        if mask.sum() == 0:
            continue
        srcs = meta_pt.loc[mask, "dataset_source"].value_counts()
        prog = scores[mask]
        print(f"  Stage {s+1} (n={mask.sum()}, score {prog.min():.3f}–{prog.max():.3f}): "
              f"{srcs.index[0]} ({srcs.iloc[0]})")

    return trans_df, stage_centroids, stage_labels


def run_transitions():
    W_norm  = np.load(f"{DATA_DIR}/W_norm.npy")
    meta_pt = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")

    trans_df, stage_centroids, stage_labels = build_transition_table(W_norm, meta_pt)

    # Save deltas for trajectory predictor
    deltas = np.array([row["delta"] for _, row in trans_df.iterrows()])
    np.save(f"{DATA_DIR}/transition_deltas.npy", deltas)
    np.save(f"{DATA_DIR}/stage_labels.npy", stage_labels)
    np.save(f"{DATA_DIR}/stage_centroids.npy",
            np.array([stage_centroids[s] for s in range(N_STAGES)]))

    trans_df.drop(columns=["delta"]).to_csv(
        f"{DATA_DIR}/transition_table.csv", index=False)
    print(f"\nSaved transition_table.csv, transition_deltas.npy")
    return trans_df, stage_centroids, stage_labels


def find_spontaneous_transitions(threshold=0.4):
    """
    Find stage transitions where one atom drops >threshold and another rises >threshold
    simultaneously — i.e. a dominant-program swap (phase transition / tipping point).
    """
    W_norm  = np.load(f"{DATA_DIR}/W_norm.npy")
    meta_pt = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")
    H_real  = np.load(f"{DATA_DIR}/H_real.npy")
    K       = H_real.shape[0]
    atom_labels = [_auto_label(H_real[k]) for k in range(K)]

    _, stage_centroids, stage_labels = build_transition_table(W_norm, meta_pt)

    print(f"\n{'='*75}")
    print(f"PHASE TRANSITIONS (tipping points, threshold={threshold})")
    print(f"{'='*75}")

    rows = []
    for s in range(N_STAGES - 1):
        c_curr = stage_centroids[s]
        c_next = stage_centroids[s + 1]
        delta  = c_next - c_curr

        rising  = [(k, delta[k]) for k in range(K) if delta[k] >  threshold]
        falling = [(k, delta[k]) for k in range(K) if delta[k] < -threshold]

        if not rising or not falling:
            continue

        rising.sort(key=lambda x: -x[1])
        falling.sort(key=lambda x: x[1])

        for (r_k, r_d), (f_k, f_d) in zip(rising, falling):
            r_lbl    = atom_labels[r_k]
            f_lbl    = atom_labels[f_k]
            r_info   = map_atom_to_diseases(r_lbl)
            f_info   = map_atom_to_diseases(f_lbl)
            r_cat    = r_info["category"]
            f_cat    = f_info["category"]
            r_dis    = ", ".join(r_info["diseases"][:2])
            f_dis    = ", ".join(f_info["diseases"][:2])

            # Derive clinical analog
            pair = frozenset([r_cat, f_cat])
            if pair == frozenset(["inflammation", "fibrosis"]):
                clinical = "Post-inflammatory fibrosis (e.g. post-COVID lung fibrosis, IBD→Crohn's fibrosis)"
            elif pair == frozenset(["cancer", "inflammation"]):
                clinical = "Tumor immune evasion via chronic inflammation (cancer-immunity cycle)"
            elif pair == frozenset(["homeostasis", "cancer"]):
                clinical = "Malignant transformation — loss of homeostatic NK/DC surveillance"
            elif pair == frozenset(["homeostasis", "inflammation"]):
                clinical = "Acute immune activation — homeostasis-to-inflammation switch"
            elif pair == frozenset(["homeostasis", "fibrosis"]):
                clinical = "Fibrotic remodeling replacing normal tissue maintenance"
            else:
                clinical = f"{f_cat.title()} → {r_cat.title()} program switch"

            print(f"\nStage {s+1}→{s+2}:")
            print(f"  FALLING: A{f_k+1} [{f_lbl}] Δ={f_d:+.3f}")
            print(f"    ↳ category: {f_cat}  |  contexts: {f_dis}")
            print(f"  RISING:  A{r_k+1} [{r_lbl}] Δ={r_d:+.3f}")
            print(f"    ↳ category: {r_cat}  |  contexts: {r_dis}")
            print(f"  CLINICAL ANALOG: {clinical}")

            rows.append({
                "stage_transition":     f"Stage {s+1}→{s+2}",
                "falling_atom":         f"A{f_k+1}",
                "falling_label":        f_lbl,
                "falling_category":     f_cat,
                "falling_delta":        round(f_d, 4),
                "falling_diseases":     f_dis,
                "rising_atom":          f"A{r_k+1}",
                "rising_label":         r_lbl,
                "rising_category":      r_cat,
                "rising_delta":         round(r_d, 4),
                "rising_diseases":      r_dis,
                "clinical_analog":      clinical,
            })

    if not rows:
        print(f"\n  (No transitions exceed threshold={threshold})")
    else:
        df = pd.DataFrame(rows)
        df.to_csv(f"{DATA_DIR}/phase_transitions.csv", index=False)
        print(f"\nSaved: {DATA_DIR}/phase_transitions.csv  ({len(df)} transitions)")
        return df

    return pd.DataFrame()


if __name__ == "__main__":
    run_transitions()
    find_spontaneous_transitions()
