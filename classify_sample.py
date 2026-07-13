"""
Step 8: Classify a new sample by projecting onto the learned NMF basis.
"""

import sys, gc, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from scipy.optimize import nnls
from liana.resource import select_resource
from pseudobulk import compute_pseudobulk, UNIVERSAL_TYPES, map_cell_type
from compute_lr_scores import compute_flow_matrix
from nmf_decompose import row_normalize
from annotate_atoms import map_atom_to_diseases, _auto_label

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
N_TYPES  = len(UNIVERSAL_TYPES)


def classify_sample(adata, cell_type_col, sample_col, H, lr_db,
                    sample_id=None, verbose=True):
    """
    Project a new sample onto the learned atom basis H.

    Parameters
    ----------
    adata        : AnnData (any annotated single-cell dataset)
    cell_type_col: obs column with cell type labels
    sample_col   : obs column with sample IDs (or None → treat as single sample)
    H            : (K, 36) NMF atom matrix from training
    lr_db        : LIANA LR database DataFrame
    sample_id    : specific sample to classify (None = first sample found)

    Returns
    -------
    dict with sample_id, w_norm (atom weights), dominant_atom, label, diseases
    """
    gene_names = adata.var_names.tolist()

    # Aggregate
    if sample_col is None:
        adata.obs["_sample"] = "s0"
        sample_col = "_sample"

    pb = compute_pseudobulk(adata, cell_type_col, sample_col)

    if sample_id is None:
        sample_id = list(pb.keys())[0]
    elif sample_id not in pb:
        available = list(pb.keys())
        print(f"  Warning: {sample_id} not found, using {available[0]}")
        sample_id = available[0]

    pb_sample = pb[sample_id]

    # Flow vector
    mat = compute_flow_matrix(pb_sample, gene_names, lr_db)
    v   = mat.flatten().astype(np.float64)

    # Normalize (same as training)
    v_norm = v / (v.sum() + 1e-12)

    # Non-negative least squares: find w s.t. H.T @ w ≈ v_norm, w >= 0
    w, residual = nnls(H.T, v_norm)

    # Normalize w to sum to 1
    w_norm = w / (w.sum() + 1e-12)

    K = H.shape[0]
    atom_labels = [_auto_label(H[k]) for k in range(K)]

    dominant_k = int(np.argmax(w_norm))
    dominant_label = atom_labels[dominant_k]
    disease_info = map_atom_to_diseases(dominant_label)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Sample: {sample_id}")
        print(f"Atom weights (sorted):")
        order = np.argsort(w_norm)[::-1]
        for rank, k in enumerate(order):
            if w_norm[k] > 0.01:
                bar = "█" * int(w_norm[k] * 40)
                print(f"  Atom {k+1:>2}: {atom_labels[k][:40]:<40} {bar} {w_norm[k]:.3f}")
        print(f"\nDominant: Atom {dominant_k+1} ({dominant_label})")
        print(f"  Category: {disease_info['category']}")
        print(f"  Clinical associations: {', '.join(disease_info['diseases'][:3])}")

    return {
        "sample_id": sample_id,
        "w_norm": w_norm,
        "dominant_atom": dominant_k + 1,
        "dominant_label": dominant_label,
        "disease_category": disease_info["category"],
        "diseases": disease_info["diseases"],
        "residual": residual,
    }


def test_classifier():
    import scanpy as sc
    from generate_synthetic import build_master_matrix

    H = np.load(f"{DATA_DIR}/H_real.npy")
    lr_db = select_resource("consensus")
    print("LIANA DB loaded.")
    print(f"H shape: {H.shape}  (K={H.shape[0]} atoms)\n")

    results = []

    # ── Test 1: pbmc3k sample (expect DC/NK atoms) ───────────────────────────
    print("=" * 60)
    print("TEST 1: pbmc3k healthy PBMC")
    adata1 = sc.datasets.pbmc3k_processed()
    adata1.obs["cell_type"] = adata1.obs["louvain"].astype(str)
    np.random.seed(7)
    adata1.obs["sample"] = np.random.choice(
        [f"pbmc3k_S{i:02d}" for i in range(10)], size=adata1.n_obs
    )
    r1 = classify_sample(adata1, "cell_type", "sample", H, lr_db,
                          sample_id="pbmc3k_S07")
    results.append(r1)
    del adata1; gc.collect()

    # ── Test 2: synthetic IPF (expect Atom 2 fibrotic) ───────────────────────
    print("\n" + "=" * 60)
    print("TEST 2: Synthetic IPF sample")
    from generate_synthetic import generate_group, base_matrix, set_edge
    import anndata as ad

    def ipf_tpl():
        m = base_matrix()
        set_edge(m, "Fibroblast", "Fibroblast", 0.95)
        set_edge(m, "Macrophage", "Fibroblast", 0.85)
        set_edge(m, "Fibroblast", "Macrophage", 0.75)
        return m

    # Build a mini synthetic AnnData with IPF-like expression
    from liana.resource import select_resource as sr
    lrdb = sr("consensus")
    real_genes = list(set(lrdb["ligand"]) | set(lrdb["receptor"]))[:500]
    n_cells, n_genes = 300, len(real_genes)
    np.random.seed(1)
    # Give Fibroblast cells high ligand expression, T_cell cells low
    labels = (["Fibroblast"] * 100 + ["Macrophage"] * 100 +
              ["T_cell"] * 50 + ["B_cell"] * 25 + ["NK_cell"] * 15 + ["Dendritic_cell"] * 10)
    X = np.random.exponential(1.0, (n_cells, n_genes)).astype(np.float32)
    # Boost Fibroblast-specific ligands (first 50 genes as proxy)
    X[:100, :50] *= 4.0   # Fibroblasts
    X[100:200, 50:100] *= 3.0   # Macrophages

    adata2 = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": labels, "sample": ["ipf_test"] * n_cells},
                         index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=real_genes),
    )
    r2 = classify_sample(adata2, "cell_type", "sample", H, lr_db,
                          sample_id="ipf_test")
    results.append(r2)
    del adata2; gc.collect()

    # ── Test 3: tumor — project pre-computed flow vector directly ────────────
    print("\n" + "=" * 60)
    print("TEST 3: Tumor TME sample (from M_real pre-computed flow)")
    from scipy.optimize import nnls as _nnls
    M_real = np.load(f"{DATA_DIR}/M_real.npy")
    meta_df = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    tumor_idx = meta_df[meta_df["dataset_source"] == "synthetic_tumor"].index[0]
    v_tumor = M_real[tumor_idx].astype(np.float64)
    v_norm  = v_tumor / (v_tumor.sum() + 1e-12)
    w_t, _  = _nnls(H.T, v_norm)
    w_norm_t = w_t / (w_t.sum() + 1e-12)

    K_loc = H.shape[0]
    from annotate_atoms import _auto_label as _al, map_atom_to_diseases as _mad
    atom_labels_loc = [_al(H[k]) for k in range(K_loc)]
    dom_t = int(np.argmax(w_norm_t))
    d_info = _mad(atom_labels_loc[dom_t])

    print(f"\n{'─'*60}")
    print(f"Sample: tumor_S01 (pre-computed flow vector)")
    print(f"Atom weights (sorted):")
    for k in np.argsort(w_norm_t)[::-1]:
        if w_norm_t[k] > 0.01:
            bar = "█" * int(w_norm_t[k] * 40)
            print(f"  Atom {k+1:>2}: {atom_labels_loc[k][:40]:<40} {bar} {w_norm_t[k]:.3f}")
    print(f"\nDominant: Atom {dom_t+1} ({atom_labels_loc[dom_t]})")
    print(f"  Category: {d_info['category']}")
    print(f"  Clinical associations: {', '.join(d_info['diseases'][:3])}")

    r3 = {
        "sample_id": "tumor_S01",
        "w_norm": w_norm_t,
        "dominant_atom": dom_t + 1,
        "dominant_label": atom_labels_loc[dom_t],
        "disease_category": d_info["category"],
        "diseases": d_info["diseases"],
        "residual": 0.0,
    }
    results.append(r3)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CLASSIFIER TEST SUMMARY")
    print(f"{'='*60}")
    expected = [
        ("pbmc3k", "homeostasis", [3, 4, 5, 6, 7]),
        ("IPF",    "fibrosis",    [2]),
        ("tumor",  "cancer",      [1]),
    ]
    for res, (name, exp_cat, exp_atoms) in zip(results, expected):
        dom = res["dominant_atom"]
        cat = res["disease_category"]
        ok_atom = dom in exp_atoms
        ok_cat  = cat == exp_cat
        print(f"\n  {name}:")
        print(f"    Dominant atom: {dom}  (expected one of {exp_atoms}) → {'✓' if ok_atom else '~'}")
        print(f"    Category: {cat}  (expected {exp_cat}) → {'✓' if ok_cat else '~'}")
        print(f"    Top disease: {res['diseases'][0]}")

    return results


if __name__ == "__main__":
    test_classifier()
