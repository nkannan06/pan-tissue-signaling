"""
Step: Statistical validation of NMF atoms.
1. Permutation test: shuffle M_real rows 1000x, compare reconstruction error to real.
2. Bootstrap stability: resample M_real 100x, cosine similarity per atom vs original.
"""

import sys, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from nmf_decompose import row_normalize, run_nmf

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"


def cosine_match(H_ref, H_new):
    """
    Greedy best-match cosine similarities between rows of H_ref and H_new.
    Returns array of length K with similarity for each ref atom's best match.
    """
    K = H_ref.shape[0]
    H_rn = H_ref / (np.linalg.norm(H_ref, axis=1, keepdims=True) + 1e-12)
    H_nn = H_new / (np.linalg.norm(H_new, axis=1, keepdims=True) + 1e-12)
    sim  = H_rn @ H_nn.T    # (K, K)

    matched, scores = set(), np.zeros(K)
    for i in range(K):
        row = sim[i].copy()
        row[list(matched)] = -1
        j = int(np.argmax(row))
        matched.add(j)
        scores[i] = sim[i, j]
    return scores


def permutation_test(M_norm, K, H_real, n_perms=1000, seed=0):
    """
    Shuffle rows of M_norm n_perms times, run NMF K, record reconstruction error.
    Returns (p_value, real_error, perm_errors).
    """
    rng = np.random.default_rng(seed)
    _, _, real_err = run_nmf(M_norm, K)

    print(f"Real reconstruction error: {real_err:.4f}")
    print(f"Running {n_perms} permutations (this takes ~2 min)...")

    perm_errors = []
    n = M_norm.shape[0]
    report_every = n_perms // 10

    for i in range(n_perms):
        M_shuf = M_norm[rng.permutation(n)]
        try:
            _, _, err = run_nmf(M_shuf, K, seed=i % 42, init="random")
            perm_errors.append(err)
        except Exception:
            pass
        if (i + 1) % report_every == 0:
            print(f"  {i+1}/{n_perms} permutations done ...")

    perm_errors = np.array(perm_errors)
    p_val = float((perm_errors <= real_err).sum()) / len(perm_errors)
    print(f"\nPermutation test:")
    print(f"  Real error          : {real_err:.4f}")
    print(f"  Permuted mean ± SD  : {perm_errors.mean():.4f} ± {perm_errors.std():.4f}")
    print(f"  Permuted range      : [{perm_errors.min():.4f}, {perm_errors.max():.4f}]")
    print(f"  p-value             : {p_val:.4f}  "
          f"({'significant' if p_val < 0.05 else 'NOT significant'} at α=0.05)")
    return p_val, real_err, perm_errors


def bootstrap_stability(M_norm, K, H_real, n_boot=100, seed=0):
    """
    Resample M_norm with replacement n_boot times, run NMF K, compute
    per-atom cosine similarity vs H_real.
    Returns DataFrame with mean ± SD cosine per atom.
    """
    rng  = np.random.default_rng(seed)
    n    = M_norm.shape[0]
    all_scores = []   # list of (K,) arrays

    print(f"\nRunning {n_boot} bootstrap resamples...")
    report_every = n_boot // 10

    for i in range(n_boot):
        idx   = rng.choice(n, size=n, replace=True)
        M_bs  = M_norm[idx]
        try:
            _, H_bs, _ = run_nmf(M_bs, K, seed=i, init="random")
            scores = cosine_match(H_real, H_bs)
            all_scores.append(scores)
        except Exception:
            pass
        if (i + 1) % report_every == 0:
            print(f"  {i+1}/{n_boot} bootstraps done ...")

    all_scores = np.array(all_scores)   # (n_boot, K)
    means = all_scores.mean(axis=0)
    sds   = all_scores.std(axis=0)

    from annotate_atoms import _auto_label
    H_real_loaded = np.load(f"{DATA_DIR}/H_real.npy")
    atom_labels   = [_auto_label(H_real_loaded[k]) for k in range(K)]

    rows = []
    print(f"\n{'─'*70}")
    print(f"{'Atom':<5} {'Label':<42} {'Cos Mean':>9} {'Cos SD':>7} {'Stable?':>8}")
    print(f"{'─'*70}")
    for k in range(K):
        stable = means[k] > 0.85
        flag   = "✓" if stable else "✗ UNSTABLE"
        print(f"{k+1:<5} {atom_labels[k][:42]:<42} {means[k]:>9.3f} {sds[k]:>7.3f} {flag:>8}")
        rows.append({
            "atom": k + 1,
            "atom_label": atom_labels[k],
            "bootstrap_cosine_mean": round(means[k], 4),
            "bootstrap_cosine_sd":   round(sds[k],   4),
            "stable": stable,
        })
    print(f"{'─'*70}")

    n_stable   = sum(r["stable"] for r in rows)
    n_unstable = K - n_stable
    print(f"\nStable atoms: {n_stable}/{K}  |  Unstable: {n_unstable}/{K}")
    return pd.DataFrame(rows), all_scores


def run_validation(n_perms=1000, n_boot=100):
    M_real = np.load(f"{DATA_DIR}/M_real.npy")
    H_real = np.load(f"{DATA_DIR}/H_real.npy")
    K      = H_real.shape[0]

    M_norm = row_normalize(M_real)

    print("=" * 70)
    print(f"STATISTICAL VALIDATION  (K={K}, n_samples={M_real.shape[0]})")
    print("=" * 70)

    # 1. Permutation test
    p_val, real_err, perm_errors = permutation_test(M_norm, K, H_real, n_perms)

    # 2. Bootstrap stability
    stab_df, all_scores = bootstrap_stability(M_norm, K, H_real, n_boot)

    # Save
    stab_df["permutation_pvalue"] = p_val
    stab_df["real_recon_error"]   = real_err
    out = f"{DATA_DIR}/validation_results.csv"
    stab_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Permutation test p-value : {p_val:.4f}  "
          f"({'PASS' if p_val < 0.05 else 'FAIL'} — real error < {100*(1-p_val):.0f}% of permuted)")
    print(f"  Bootstrap stable atoms   : {stab_df['stable'].sum()}/{K}")
    unstable = stab_df[~stab_df["stable"]]
    if len(unstable):
        for _, row in unstable.iterrows():
            print(f"    Unstable: Atom {row['atom']} ({row['atom_label'][:40]}) "
                  f"cos={row['bootstrap_cosine_mean']:.3f}±{row['bootstrap_cosine_sd']:.3f}")

    return p_val, stab_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--perms",  type=int, default=1000)
    parser.add_argument("--boots",  type=int, default=100)
    parser.add_argument("--fast",   action="store_true",
                        help="Quick mode: 100 perms, 20 bootstraps")
    args = parser.parse_args()

    if args.fast:
        args.perms, args.boots = 100, 20

    run_validation(n_perms=args.perms, n_boot=args.boots)
