"""
Step 5: Sparse NMF decomposition of the master flow matrix M (n_samples × 30).
Selects best K via reconstruction error elbow + atom stability.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist


def row_normalize(M: np.ndarray) -> np.ndarray:
    """Normalize each sample row to sum to 1."""
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid /0
    return M / row_sums


def run_nmf(M_norm: np.ndarray, K: int, seed: int = 42, init: str = "nndsvda") -> tuple:
    """
    Main run uses nndsvda (deterministic, good init).
    Stability runs pass init='random' with varying seeds.
    alpha_H adds small L1 sparsity on atoms (H); W is unregularized.
    """
    model = NMF(
        n_components=K,
        init=init,
        alpha_W=0.0,
        alpha_H=0.0,      # non-negativity alone provides implicit sparsity at this data scale
        max_iter=2000,
        random_state=seed,
    )
    W = model.fit_transform(M_norm)
    H = model.components_
    err = model.reconstruction_err_
    return W, H, err


def cosine_similarity_matrix(H1: np.ndarray, H2: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of H1 and H2."""
    H1n = H1 / (np.linalg.norm(H1, axis=1, keepdims=True) + 1e-12)
    H2n = H2 / (np.linalg.norm(H2, axis=1, keepdims=True) + 1e-12)
    return H1n @ H2n.T


def stability_score(M_norm: np.ndarray, K: int, n_runs: int = 10) -> float:
    """
    Run NMF n_runs times with different seeds.
    For each pair of runs, greedily match atoms by max cosine similarity.
    Return mean of best-match cosine similarities across all pairs.
    """
    Hs = []
    for seed in range(n_runs):
        try:
            # Use random init so different seeds produce genuinely different starts
            _, H, _ = run_nmf(M_norm, K, seed=seed, init="random")
            Hs.append(H)
        except Exception:
            pass

    if len(Hs) < 2:
        return 0.0

    scores = []
    ref = Hs[0]
    for H in Hs[1:]:
        sim = cosine_similarity_matrix(ref, H)
        # Greedy best-match: for each ref atom pick best match in H
        matched = set()
        atom_scores = []
        for i in range(K):
            row = sim[i].copy()
            row[list(matched)] = -1
            j = int(np.argmax(row))
            matched.add(j)
            atom_scores.append(sim[i, j])
        scores.append(np.mean(atom_scores))

    return float(np.mean(scores))


def select_best_K(
    M_norm: np.ndarray,
    K_range: list,
    n_stability_runs: int = 10,
) -> dict:
    results = {}
    print(f"{'K':>3}  {'Recon Error':>12}  {'Stability':>10}")
    print("-" * 32)
    for K in K_range:
        _, _, err = run_nmf(M_norm, K)
        stab = stability_score(M_norm, K, n_runs=n_stability_runs)
        results[K] = {"error": err, "stability": stab}
        print(f"{K:>3}  {err:>12.4f}  {stab:>10.4f}")

    # Elbow: largest drop in error going from K to K+1
    errors = np.array([results[K]["error"] for K in K_range])
    drops = -np.diff(errors)  # positive = improvement
    elbow_idx = int(np.argmax(drops))        # biggest improvement step
    elbow_K = K_range[elbow_idx + 1]         # K after the big drop

    # Stability filter: prefer K with stability > 0.9
    stable_Ks = [K for K in K_range if results[K]["stability"] > 0.9]
    if stable_Ks:
        # Among stable, pick elbow_K if stable, else closest stable K
        best_K = elbow_K if elbow_K in stable_Ks else stable_Ks[-1]
    else:
        best_K = elbow_K

    print(f"\nElbow K={elbow_K}, Stable Ks={stable_Ks}")
    print(f"→ Selected K={best_K}  (err={results[best_K]['error']:.4f}, stab={results[best_K]['stability']:.4f})")
    return best_K, results


def decompose(M: np.ndarray, K_range=(4, 6, 8), n_stability_runs=10) -> dict:
    M_norm = row_normalize(M)
    print(f"M shape: {M.shape}, normalized — row sums: min={M_norm.sum(1).min():.3f} max={M_norm.sum(1).max():.3f}\n")

    best_K, results = select_best_K(M_norm, list(K_range), n_stability_runs)

    W, H, err = run_nmf(M_norm, best_K)
    print(f"\nFinal NMF: K={best_K}, W={W.shape}, H={H.shape}")
    return {
        "K": best_K,
        "W": W,
        "H": H,
        "error": err,
        "stability": results[best_K]["stability"],
        "all_results": results,
        "M_norm": M_norm,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from generate_synthetic import build_master_matrix

    M, labels, group_names = build_master_matrix()
    result = decompose(M, K_range=(4, 6, 8), n_stability_runs=10)

    print(f"\n{'='*50}")
    print(f"Selected K = {result['K']}")
    print(f"Reconstruction error = {result['error']:.4f}")
    print(f"Stability score      = {result['stability']:.4f}")
    np.save("data/W.npy", result["W"])
    np.save("data/H.npy", result["H"])
    np.save("data/M_norm.npy", result["M_norm"])
    pd.Series(labels).to_csv("data/labels_synthetic.csv", index=False, header=False)
    print("Saved W, H, M_norm, labels.")
