"""
Step 3: Compute 6×6 sender-receiver flow matrix per sample.
score(sender→receiver) = Σ over top-50 LR pairs of expr[sender][L] * expr[receiver][R]
"""

import gc
import numpy as np
import pandas as pd
from liana.resource import select_resource

UNIVERSAL_TYPES = ["Macrophage", "T_cell", "B_cell", "NK_cell", "Dendritic_cell", "Fibroblast"]
N_TOP_LR = 50


def load_lr_database() -> pd.DataFrame:
    """Load LIANA consensus LR pairs. Returns DataFrame with columns [ligand, receptor]."""
    lr_db = select_resource("consensus")
    print(f"LIANA consensus DB: {len(lr_db)} LR pairs")
    return lr_db


def compute_flow_matrix(pb_sample: dict, gene_names: list, lr_db: pd.DataFrame) -> np.ndarray:
    """
    Parameters
    ----------
    pb_sample  : {cell_type: mean_expr_array}  for one sample
    gene_names : list of gene names (matches array positions)
    lr_db      : DataFrame with columns [ligand, receptor]

    Returns
    -------
    flow : (6, 6) float32 ndarray  [sender_i, receiver_j]
    """
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    # Filter LR pairs to genes present in this dataset
    lr_present = lr_db[
        lr_db["ligand"].isin(gene_idx) & lr_db["receptor"].isin(gene_idx)
    ].copy()

    if len(lr_present) == 0:
        print("  WARNING: no LR pairs overlap with gene set — returning zeros")
        return np.zeros((6, 6), dtype=np.float32)

    # Pre-index gene expression arrays for speed
    type_expr = {ct: pb_sample[ct] for ct in UNIVERSAL_TYPES}

    n = len(UNIVERSAL_TYPES)
    flow = np.zeros((n, n), dtype=np.float32)

    for i, sender in enumerate(UNIVERSAL_TYPES):
        L_expr = type_expr[sender]
        for j, receiver in enumerate(UNIVERSAL_TYPES):
            R_expr = type_expr[receiver]

            # Vectorized: score each LR pair
            L_vals = L_expr[[gene_idx[g] for g in lr_present["ligand"]]]
            R_vals = R_expr[[gene_idx[g] for g in lr_present["receptor"]]]
            pair_scores = L_vals * R_vals

            # Top-N sum
            if len(pair_scores) >= N_TOP_LR:
                top_idx = np.argpartition(pair_scores, -N_TOP_LR)[-N_TOP_LR:]
                flow[i, j] = pair_scores[top_idx].sum()
            else:
                flow[i, j] = pair_scores.sum()

    return flow


def compute_all_flow_matrices(pb: dict, gene_names: list, lr_db: pd.DataFrame) -> dict:
    """
    Returns {sample_id: (6,6) flow matrix}.
    """
    matrices = {}
    for sid, pb_sample in pb.items():
        mat = compute_flow_matrix(pb_sample, gene_names, lr_db)
        matrices[sid] = mat
        print(f"  {sid}: min={mat.min():.3f}, max={mat.max():.3f}, sum={mat.sum():.3f}")

    return matrices


def print_flow_matrix(mat: np.ndarray, sample_id: str = ""):
    """Pretty-print a 6×6 flow matrix."""
    short = [t[:4] for t in UNIVERSAL_TYPES]
    df = pd.DataFrame(mat, index=UNIVERSAL_TYPES, columns=UNIVERSAL_TYPES)
    print(f"\n--- Flow matrix {sample_id} (sender rows → receiver cols) ---")
    print(df.round(2).to_string())
    print(f"Non-zero entries: {(mat > 0).sum()}/36")


# ── unit test with synthetic pseudobulk ────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    n_genes = 500
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    # Fake pseudobulk: random expression
    pb_sample = {ct: np.random.exponential(1.0, n_genes).astype(np.float32)
                 for ct in UNIVERSAL_TYPES}

    lr_db = load_lr_database()

    # Inject some real gene names so a few LR pairs match
    real_genes = list(set(lr_db["ligand"].tolist() + lr_db["receptor"].tolist()))[:100]
    gene_names[:100] = real_genes
    for ct in UNIVERSAL_TYPES:
        # Give those positions non-zero expression
        pb_sample[ct][:100] = np.random.exponential(2.0, 100).astype(np.float32)

    mat = compute_flow_matrix(pb_sample, gene_names, lr_db)
    print_flow_matrix(mat, sample_id="synthetic_test")

    is_nonzero = (mat > 0).any()
    print(f"\nMatrix is non-zero: {is_nonzero}")
    assert is_nonzero, "Flow matrix is all zeros — check gene name overlap!"
    print("compute_lr_scores unit test PASSED")
