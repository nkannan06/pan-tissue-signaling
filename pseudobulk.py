"""
Step 2: Pseudobulk aggregation.
Maps heterogeneous cell type labels → universal 6-type vocabulary via fuzzy matching.
Returns {sample_id: {cell_type: mean_expr_array}}.
"""

import gc
import re
from typing import Optional
import numpy as np
import scipy.sparse as sp

UNIVERSAL_TYPES = ["Macrophage", "T_cell", "B_cell", "NK_cell", "Dendritic_cell", "Fibroblast"]

# Keyword patterns → canonical name (order matters: more specific first)
_PATTERNS = [
    (r"dendritic|dendr|pdc|mdc|\bdc\b|cd11c", "Dendritic_cell"),
    (r"nk.?cells?|natural.?killer|\bnk\b|cd56\+|killer", "NK_cell"),
    (r"macrophage|macro|kupffer|microglia|\bmac\b|monocyte|mono\b|cd14\+|cd16\+|fcgr3a", "Macrophage"),
    (r"t.?cells?|tcells?|cd[348]\+|cytotoxic|helper|regulatory t|treg|effector t|naive t|memory t|cd4\b|cd8\b", "T_cell"),
    (r"b.?cells?|bcells?|plasma.?cells?|plasmablast|b lymph|cd19\+|cd20\+|\bcb\b", "B_cell"),
    (r"fibroblast|fibro|stellate|mesenchymal|stroma", "Fibroblast"),
]


def map_cell_type(label: str) -> Optional[str]:
    """Return canonical type or None if no match."""
    s = label.lower().strip()
    for pattern, canonical in _PATTERNS:
        if re.search(pattern, s):
            return canonical
    return None


def compute_pseudobulk(adata, cell_type_col: str, sample_col: str) -> dict:
    """
    Parameters
    ----------
    adata         : AnnData with raw/normalized counts in .X
    cell_type_col : obs column containing cell type labels
    sample_col    : obs column containing sample/donor IDs

    Returns
    -------
    pb : dict  {sample_id: {cell_type: np.ndarray (n_genes,)}}
    """
    import scanpy as sc

    # Normalize if not already (log1p scale)
    if adata.X.max() > 50:
        print("Normalizing counts (total → 1e4 → log1p)...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    obs = adata.obs[[cell_type_col, sample_col]].copy()
    obs["universal"] = obs[cell_type_col].map(map_cell_type)

    # Report mapping
    print("\n--- Cell type mapping ---")
    mapping_counts = (
        obs.groupby([cell_type_col, "universal"]).size().reset_index(name="n")
    )
    for _, row in mapping_counts.iterrows():
        tag = row["universal"] if row["universal"] else "UNMAPPED"
        print(f"  {row[cell_type_col]:<35} → {tag:<20} (n={row['n']})")

    unmapped_n = obs["universal"].isna().sum()
    total_n = len(obs)
    print(f"\nMapped {total_n - unmapped_n}/{total_n} cells "
          f"({100*(total_n-unmapped_n)/total_n:.1f}%)")

    # Filter to mapped cells only
    obs = obs[obs["universal"].notna()]

    pb = {}
    X = adata.X  # keep reference; don't copy full matrix

    samples = obs[sample_col].unique()
    print(f"\nAggregating {len(samples)} samples × {len(UNIVERSAL_TYPES)} cell types ...")

    for sid in samples:
        sid_mask = obs[sample_col] == sid
        pb[sid] = {}
        for ctype in UNIVERSAL_TYPES:
            ct_mask = sid_mask & (obs["universal"] == ctype)
            idx = np.where(ct_mask)[0]
            if len(idx) == 0:
                # zero vector placeholder
                pb[sid][ctype] = np.zeros(adata.n_vars, dtype=np.float32)
                continue
            block = X[idx]
            if sp.issparse(block):
                mean_vec = np.asarray(block.mean(axis=0)).flatten().astype(np.float32)
            else:
                mean_vec = block.mean(axis=0).astype(np.float32)
            pb[sid][ctype] = np.maximum(mean_vec, 0)

        # per-sample cell-type counts
        counts = {ct: int((sid_mask & (obs["universal"] == ct)).sum()) for ct in UNIVERSAL_TYPES}
        print(f"  {sid}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # Free obs copy
    del obs
    gc.collect()

    return pb


# ── quick unit test with synthetic data ────────────────────────────────────────
if __name__ == "__main__":
    import anndata as ad
    import pandas as pd

    np.random.seed(42)
    n_cells, n_genes = 600, 200
    labels = (
        ["Macrophage"] * 100 + ["T cell"] * 100 + ["B cell"] * 100 +
        ["NK cell"] * 100 + ["Dendritic cell"] * 100 + ["Fibroblast"] * 100
    )
    samples = ["S1"] * 300 + ["S2"] * 300
    X = np.random.poisson(3, size=(n_cells, n_genes)).astype(np.float32)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": labels, "sample": samples}),
        var=pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)]),
    )

    pb = compute_pseudobulk(adata, cell_type_col="cell_type", sample_col="sample")
    print(f"\nResult keys: samples={list(pb.keys())}, types={list(pb['S1'].keys())}")
    print(f"Mean expr array shape: {pb['S1']['Macrophage'].shape}")
    print("Pseudobulk unit test PASSED")
