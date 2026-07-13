"""
Task C: Novelty metrics for the paper.
1. Cross-tissue atom conservation (cosine similarity real vs synthetic)
2. Classifier accuracy via 5-fold cross-validation
3. Trajectory prediction error (W-space euclidean)
4. Intervention specificity
"""

import sys, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import nnls

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from annotate_atoms import _auto_label, map_atom_to_diseases
from nmf_decompose import row_normalize

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
N_STAGES = 5
SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def cosine_sim_vectors(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def greedy_match(H_ref, H_new):
    """Greedy cosine match: returns (matched_sim_array, perm_ref→new)."""
    K = H_ref.shape[0]
    used = set()
    sims = []
    for k in range(K):
        best_j, best_s = -1, -1.0
        for j in range(H_new.shape[0]):
            if j in used:
                continue
            s = cosine_sim_vectors(H_ref[k], H_new[j])
            if s > best_s:
                best_s, best_j = s, j
        used.add(best_j)
        sims.append(best_s)
    return np.array(sims)


def nmf_fit(M, K, seed=SEED):
    model = NMF(n_components=K, init="nndsvda", random_state=seed,
                max_iter=500, alpha_W=0, alpha_H=0)
    W = model.fit_transform(M)
    H = model.components_
    return W, H


def project_sample(v_norm, H):
    """NNLS projection of a single normalized flow vector onto atom basis."""
    w, _ = nnls(H.T, v_norm)
    s = w.sum()
    return w / (s + 1e-12) if s > 1e-12 else w


# ── Metric 1: Cross-tissue atom conservation ───────────────────────────────────

def metric1_atom_conservation():
    H_real  = np.load(f"{DATA_DIR}/H_real.npy")
    H_synth = np.load(f"{DATA_DIR}/H.npy")
    sims    = greedy_match(H_real[:2], H_synth)   # atoms 1 & 2 per summary
    # Full match for all K
    sims_all = greedy_match(H_real, H_synth)
    print("\n─── Metric 1: Cross-Tissue Atom Conservation ───────────────────")
    print(f"  Atom 1 (Cancer/Mac→Tcell):   cosine sim = {sims[0]:.3f}")
    print(f"  Atom 2 (Fibrosis/Fibro→Fibro): cosine sim = {sims[1]:.3f}")
    print(f"  Mean all atoms:               cosine sim = {sims_all.mean():.3f} ± {sims_all.std():.3f}")
    return sims_all


# ── Metric 2: 5-fold cross-validation classifier accuracy ─────────────────────

def metric2_classifier_accuracy():
    M_real   = np.load(f"{DATA_DIR}/M_real.npy")
    meta     = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    M_norm   = row_normalize(M_real)
    n        = M_norm.shape[0]
    K        = np.load(f"{DATA_DIR}/H_real.npy").shape[0]
    np.random.seed(SEED)
    perm     = np.random.permutation(n)
    n_folds  = 5
    fold_sz  = n // n_folds

    # Ground truth: dominant expected category from dataset_source
    SRC_CAT = {
        "pbmc3k":"homeostasis","pbmc68k":"homeostasis",
        "synthetic_covid":"inflammation","synthetic_ipf":"fibrosis",
        "synthetic_tumor":"cancer",
    }
    gt_cats = [SRC_CAT[s] for s in meta["dataset_source"]]

    accs = []
    print("\n─── Metric 2: 5-Fold Cross-Validation Classifier Accuracy ──────")
    for fold in range(n_folds):
        val_idx   = perm[fold * fold_sz : (fold + 1) * fold_sz]
        train_idx = np.concatenate([perm[:fold*fold_sz], perm[(fold+1)*fold_sz:]])

        M_train = M_norm[train_idx]
        M_val   = M_norm[val_idx]

        _, H_fold = nmf_fit(M_train, K)

        correct = 0
        for vi in range(len(val_idx)):
            w_pred = project_sample(M_val[vi], H_fold)
            # Dominant atom's category
            dom_k = int(np.argmax(w_pred))
            H_full = np.load(f"{DATA_DIR}/H_real.npy")
            H_match = greedy_match(H_full, H_fold)
            # Which full atom does fold atom dom_k match best?
            best_real_k = int(np.argmax(
                [cosine_sim_vectors(H_full[rk], H_fold[dom_k]) for rk in range(K)]
            ))
            pred_lbl  = _auto_label(H_full[best_real_k])
            pred_info = map_atom_to_diseases(pred_lbl)
            pred_cat  = pred_info["category"]
            true_cat  = gt_cats[val_idx[vi]]
            if pred_cat == true_cat:
                correct += 1

        acc = correct / len(val_idx)
        accs.append(acc)
        print(f"  Fold {fold+1}: {correct}/{len(val_idx)} correct  →  accuracy = {acc:.2%}")

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    print(f"  Mean accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    return accs


# ── Metric 3: Trajectory prediction error ─────────────────────────────────────

def metric3_trajectory_error():
    """
    For each sample in stage s, apply one prediction step and compare
    to the ACTUAL mean W of stage s+1 (the ground-truth next state).
    Note: delta-based trajectory is by definition unbiased on centroids;
    this metric captures sample-level spread around that prediction.
    """
    W_norm          = np.load(f"{DATA_DIR}/W_norm.npy")
    meta_pt         = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")
    stage_centroids = np.load(f"{DATA_DIR}/stage_centroids.npy")
    deltas          = np.load(f"{DATA_DIR}/transition_deltas.npy")
    stage_labels    = np.load(f"{DATA_DIR}/stage_labels.npy")

    from predict_trajectory import apply_transition

    print("\n─── Metric 3: Trajectory Prediction Error (per-sample, W-space L2) ─")
    print("  (each sample in stage s predicted→next, vs actual stage s+1 centroid)")

    all_errors = []
    for s in range(N_STAGES - 1):
        curr_mask = stage_labels == s
        next_mask = stage_labels == s + 1
        if not curr_mask.any() or not next_mask.any():
            continue

        actual_next = W_norm[next_mask].mean(axis=0)
        stage_errs  = []
        for w_sample in W_norm[curr_mask]:
            pred = apply_transition(w_sample, deltas, s, n_steps=1)[-1]
            err  = float(np.linalg.norm(pred - actual_next))
            stage_errs.append(err)

        mean_s = float(np.mean(stage_errs))
        std_s  = float(np.std(stage_errs))
        all_errors.extend(stage_errs)
        print(f"  Stage {s+1}→{s+2}  (n={curr_mask.sum()} samples):  "
              f"mean L2 = {mean_s:.4f} ± {std_s:.4f}")

    mean_err = float(np.mean(all_errors))
    print(f"  Overall mean prediction error: {mean_err:.4f}")
    return all_errors


# ── Metric 4: Intervention specificity ────────────────────────────────────────

def metric4_intervention_specificity():
    """
    For each intervention, compute:
      specificity = target_delta / mean(off_target_deltas)
    Ideal: target changes a lot, off-targets change < 10% as much.
    """
    from simulate_intervention import apply_intervention, INTERVENTIONS
    W_norm          = np.load(f"{DATA_DIR}/W_norm.npy")
    meta_pt         = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")
    H_real          = np.load(f"{DATA_DIR}/H_real.npy")
    K               = H_real.shape[0]

    # Use the IPF sample as test case (highest fibrosis loading)
    ipf_mask = meta_pt["dataset_source"] == "synthetic_ipf"
    ipf_sub  = meta_pt[ipf_mask].sort_values("progression_score")
    ipf_idx  = ipf_sub.index[-1]
    w        = W_norm[ipf_idx]

    print("\n─── Metric 4: Intervention Specificity ─────────────────────────")
    print(f"  Test sample: {meta_pt.iloc[ipf_idx]['sample_id']}  "
          f"(progression={meta_pt.iloc[ipf_idx]['progression_score']:.3f})")

    results = []
    for name, atom_scales in INTERVENTIONS.items():
        w_tx = apply_intervention(w, atom_scales)
        raw_deltas = np.abs(w_tx - w)

        target_delta    = float(np.mean([raw_deltas[k] for k in atom_scales]))
        off_target_idxs = [k for k in range(K) if k not in atom_scales]
        off_target_mean = float(raw_deltas[off_target_idxs].mean()) if off_target_idxs else 0.0
        specificity     = target_delta / (off_target_mean + 1e-9)

        clean = name.replace("\n", " ")
        print(f"  {clean}")
        print(f"    target_delta={target_delta:.4f}  off_target_mean={off_target_mean:.4f}  "
              f"specificity={specificity:.1f}x")
        results.append({"intervention": clean, "target_delta": round(target_delta,4),
                        "off_target_mean": round(off_target_mean,4),
                        "specificity_ratio": round(specificity,2)})

    return pd.DataFrame(results)


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(sims_all, accs, errors, spec_df):
    print("\n" + "═"*62)
    print("NOVELTY METRICS SUMMARY (for paper results section)")
    print("═"*62)
    print(f"  1. Atom conservation  — A1 (Cancer):   {sims_all[0]:.3f}")
    print(f"                        — A2 (Fibrosis): {sims_all[1]:.3f}")
    print(f"                        — Mean all K:    {sims_all.mean():.3f} ± {sims_all.std():.3f}")
    print(f"  2. Classifier 5-fold  — Mean accuracy: {np.mean(accs):.2%} ± {np.std(accs):.2%}")
    print(f"  3. Trajectory error   — Mean L2 (W):   {np.mean(errors):.4f}")
    print(f"  4. Specificity ratio  (higher = more targeted intervention):")
    for _, row in spec_df.iterrows():
        print(f"       {row['intervention']:<45} {row['specificity_ratio']:.1f}x")
    print("═"*62)


if __name__ == "__main__":
    sims_all = metric1_atom_conservation()
    accs     = metric2_classifier_accuracy()
    errors   = metric3_trajectory_error()
    spec_df  = metric4_intervention_specificity()
    print_summary(sims_all, accs, errors, spec_df)
