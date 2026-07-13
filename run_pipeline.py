"""
Step 4: End-to-end test pipeline.
Tries to load GSE154826 from GEO. Falls back to synthetic data if download fails.
Prints RAM usage, 6×6 flow matrix, and confirms non-zero.
"""

import gc
import os
import sys
import psutil
import numpy as np

sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")
from pseudobulk import compute_pseudobulk, UNIVERSAL_TYPES
from compute_lr_scores import load_lr_database, compute_all_flow_matrices, print_flow_matrix

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data/GSE154826"


def ram_gb():
    return psutil.virtual_memory().used / 1e9


def try_load_gse154826():
    """
    Attempt to download + load GSE154826 from GEO.
    Returns (adata, cell_type_col, sample_col) or None on failure.
    """
    import urllib.request
    import tarfile
    import scanpy as sc
    import anndata as ad

    os.makedirs(DATA_DIR, exist_ok=True)

    # GSE154826 supplementary: one processed h5 per sample (10x format)
    # The RAW tar is ~2GB — try the soft file first to find smaller per-sample files
    soft_url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/"
                "GSE154826/soft/GSE154826_family.soft.gz")
    soft_path = os.path.join(DATA_DIR, "GSE154826_family.soft.gz")

    print("Fetching GEO soft file for file list...")
    try:
        urllib.request.urlretrieve(soft_url, soft_path)
    except Exception as e:
        print(f"Soft file download failed: {e}")
        return None

    # Parse soft to find supplementary files
    import gzip
    suppl_urls = []
    with gzip.open(soft_path, "rt", errors="ignore") as f:
        for line in f:
            if "supplementary_file" in line.lower() and "ftp" in line.lower():
                url = line.strip().split("=")[-1].strip()
                suppl_urls.append(url)

    print(f"Found {len(suppl_urls)} supplementary files:")
    for u in suppl_urls[:10]:
        print(f"  {u}")

    # Look for a single processed matrix (h5 or mtx tar)
    h5_urls = [u for u in suppl_urls if u.endswith(".h5") or "filtered" in u.lower()]
    tar_urls = [u for u in suppl_urls if u.endswith(".tar")]

    target_url = None
    if h5_urls:
        target_url = h5_urls[0]
    elif tar_urls:
        target_url = tar_urls[0]

    if not target_url:
        print("No suitable supplementary file found.")
        return None

    fname = os.path.basename(target_url)
    fpath = os.path.join(DATA_DIR, fname)

    if not os.path.exists(fpath):
        print(f"\nDownloading {target_url} ...")
        try:
            urllib.request.urlretrieve(target_url, fpath)
            print(f"Saved to {fpath} ({os.path.getsize(fpath)/1e6:.1f} MB)")
        except Exception as e:
            print(f"Download failed: {e}")
            return None

    # Try loading
    try:
        if fpath.endswith(".h5"):
            adata = sc.read_10x_h5(fpath)
        elif fpath.endswith(".tar"):
            extract_dir = os.path.join(DATA_DIR, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(fpath) as tf:
                tf.extractall(extract_dir)
            # find mtx dir
            for root, dirs, files in os.walk(extract_dir):
                if any("matrix.mtx" in f for f in files):
                    adata = sc.read_10x_mtx(root, var_names="gene_symbols")
                    break
            else:
                return None
        else:
            return None

        print(f"Loaded adata: {adata.shape}")
        return adata, None, None  # cell_type/sample cols unknown until we check obs
    except Exception as e:
        print(f"Load failed: {e}")
        return None


def make_synthetic_adata():
    """
    Realistic synthetic dataset: 1200 cells × 2000 genes, 2 samples, 6 cell types.
    Injects real LIANA gene names so LR scoring is non-trivial.
    """
    import anndata as ad
    import pandas as pd
    from liana.resource import select_resource

    print("Building synthetic AnnData with real LIANA gene names...")
    lr_db = select_resource("consensus")
    real_genes = list(
        set(lr_db["ligand"].tolist() + lr_db["receptor"].tolist())
    )[:1000]
    n_extra = max(0, 2000 - len(real_genes))
    gene_names = real_genes + [f"EXTRA{i}" for i in range(n_extra)]
    n_genes = len(gene_names)

    np.random.seed(42)
    n_per_type = 100
    n_samples = 2
    cell_types_raw = UNIVERSAL_TYPES * n_per_type * n_samples
    n_cells = len(cell_types_raw)
    sample_labels = (["Sample1"] * (n_cells // 2) + ["Sample2"] * (n_cells // 2))

    # Simulate count-like expression (Poisson)
    X = np.random.poisson(
        lam=np.random.exponential(1.5, size=(1, n_genes)),
        size=(n_cells, n_genes),
    ).astype(np.float32)

    # Boost ligand/receptor genes so LR signal is detectable
    lr_gene_idx = list(range(len(real_genes)))
    X[:, lr_gene_idx] += np.random.exponential(2.0, size=(n_cells, len(lr_gene_idx)))

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cell_type": cell_types_raw, "sample": sample_labels},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=gene_names),
    )
    print(f"Synthetic adata: {adata.shape}")
    return adata, "cell_type", "sample"


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PAN-TISSUE SIGNALING — end-to-end pipeline test")
    print("=" * 60)
    print(f"\nRAM before: {ram_gb():.2f} GB")

    # Step 1: load data
    result = try_load_gse154826()

    if result is None:
        print("\nGEO download failed — using synthetic data.\n")
        adata, ct_col, samp_col = make_synthetic_adata()
    else:
        adata, ct_col, samp_col = result
        if ct_col is None:
            print("\nAdata loaded but cell type annotation unknown — switching to synthetic.")
            del adata
            gc.collect()
            adata, ct_col, samp_col = make_synthetic_adata()

    print(f"RAM after data load: {ram_gb():.2f} GB")

    # Step 2: pseudobulk
    print("\n--- Pseudobulk aggregation ---")
    gene_names = adata.var_names.tolist()
    pb = compute_pseudobulk(adata, cell_type_col=ct_col, sample_col=samp_col)

    del adata
    gc.collect()
    print(f"RAM after pseudobulk (adata deleted): {ram_gb():.2f} GB")

    # Step 3: LR flow matrices
    print("\n--- Computing LR flow matrices ---")
    lr_db = load_lr_database()
    matrices = compute_all_flow_matrices(pb, gene_names, lr_db)

    # Step 4: Report
    first_sid = list(matrices.keys())[0]
    mat = matrices[first_sid]
    print_flow_matrix(mat, sample_id=first_sid)

    is_nonzero = (mat > 0).any()
    print(f"\nMatrix non-zero: {is_nonzero}")
    print(f"RAM at end: {ram_gb():.2f} GB")

    if is_nonzero:
        print("\nPIPELINE TEST PASSED — ready to scale up.")
    else:
        print("\nWARNING: flow matrix is all zeros — check gene name overlap.")
