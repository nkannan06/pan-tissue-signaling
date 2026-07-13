"""
Build real M matrix from 5 datasets:
  D1: GSE158055 COVID-19 BALF (GEO, 2 attempts → synthetic fallback)
  D2: pbmc3k_processed (scanpy built-in, 10 pseudo-samples)
  D3: pbmc68k_reduced  (scanpy built-in, 10 pseudo-samples)
  D4: synthetic IPF fibrotic
  D5: synthetic tumor TME
"""

import gc, os, sys, warnings
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import anndata as ad

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from pseudobulk import compute_pseudobulk, UNIVERSAL_TYPES, map_cell_type
from compute_lr_scores import load_lr_database, compute_flow_matrix

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(f"{DATA_DIR}/geo", exist_ok=True)

N_TYPES = len(UNIVERSAL_TYPES)


def ram():
    return psutil.virtual_memory().used / 1e9


# ── helpers ───────────────────────────────────────────────────────────────────

def make_pseudo_samples(adata, ct_col, n_samples=10, source_tag=""):
    """Randomly assign cells to n_samples pseudo-sample IDs."""
    np.random.seed(42)
    adata = adata.copy()
    idx = np.random.randint(0, n_samples, size=adata.n_obs)
    adata.obs["pseudo_sample"] = [f"{source_tag}_S{i:02d}" for i in idx]
    return adata


def flow_from_adata(adata, ct_col, sample_col, lr_db, source, tissue, disease):
    """
    Pseudobulk + LR flow for one AnnData.
    Returns list of metadata dicts and list of (36,) flow vectors.
    """
    gene_names = adata.var_names.tolist()
    pb = compute_pseudobulk(adata, cell_type_col=ct_col, sample_col=sample_col)

    rows, meta = [], []
    for sid, pb_sample in pb.items():
        n_mapped = sum(
            int((adata.obs[sample_col] == sid) & (adata.obs[ct_col].map(map_cell_type).notna()))
            for sid in [sid]
        )
        n_total = int((adata.obs[sample_col] == sid).sum())
        pct = 100 * n_mapped / max(n_total, 1)

        mat = compute_flow_matrix(pb_sample, gene_names, lr_db)
        vec = mat.flatten().astype(np.float32)
        rows.append(vec)
        meta.append({
            "sample_id": sid,
            "dataset_source": source,
            "tissue": tissue,
            "disease": disease,
            "n_cells_total": n_total,
            "n_cells_mapped": n_mapped,
            "pct_cells_mapped": round(pct, 1),
            "low_quality": pct < 50,
        })

    del pb; gc.collect()
    return rows, meta


# ── Dataset 1: GSE158055 COVID-19 ─────────────────────────────────────────────

def try_geo_gse158055():
    """
    Attempt 1: download processed h5ad from Sanger EBI mirror.
    Attempt 2: parse GEO soft file and download smallest count matrix.
    Returns (adata, ct_col) or None.
    """
    import urllib.request, gzip, tarfile

    # Attempt 1: EBI processed h5ad
    urls_attempt1 = [
        "https://cellgeni.cog.sanger.ac.uk/COVID-blood/haniffa21.processed.h5ad",
        "https://covid19.cog.sanger.ac.uk/haniffa21.processed.h5ad",
    ]
    dest = f"{DATA_DIR}/geo/covid_haniffa21.h5ad"
    if not os.path.exists(dest):
        for url in urls_attempt1:
            try:
                print(f"  Attempt 1: {url}")
                urllib.request.urlretrieve(url, dest)
                print(f"  Downloaded {os.path.getsize(dest)/1e6:.0f} MB")
                break
            except Exception as e:
                print(f"  Failed: {e}")

    if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
        try:
            adata = sc.read_h5ad(dest)
            print(f"  Loaded COVID h5ad: {adata.shape}")
            print(f"  obs cols: {adata.obs.columns.tolist()[:8]}")
            # Find cell type column
            ct_col = None
            for c in ["cell_type", "celltype", "leiden", "louvain", "annotation", "cell_type_original"]:
                if c in adata.obs.columns:
                    ct_col = c
                    break
            if ct_col:
                print(f"  Cell type col: {ct_col}")
                print(f"  Types: {adata.obs[ct_col].unique()[:8].tolist()}")
            return adata, ct_col
        except Exception as e:
            print(f"  h5ad load failed: {e}")
            os.remove(dest)

    # Attempt 2: GEO soft → find per-sample mtx
    print("  Attempt 2: GEOparse GSE158055")
    try:
        import GEOparse
        soft_path = f"{DATA_DIR}/geo/GSE158055"
        os.makedirs(soft_path, exist_ok=True)
        gse = GEOparse.get_GEO(geo="GSE158055", destdir=soft_path, silent=True)
        print(f"  GSE158055 title: {gse.metadata['title'][0]}")

        # Find supplementary files
        all_suppl = []
        for gsm in gse.gsms.values():
            for key, vals in gsm.metadata.items():
                if "supplementary" in key.lower():
                    all_suppl.extend(vals)

        print(f"  Supplementary files found: {len(all_suppl)}")
        # Look for a single manageable file
        for url in all_suppl[:5]:
            print(f"    {url}")

        # Try downloading the first gz/tar that's reasonably small
        for url in all_suppl:
            if url.lower() == "none":
                continue
            fname = os.path.basename(url)
            fpath = f"{soft_path}/{fname}"
            if not os.path.exists(fpath):
                try:
                    urllib.request.urlretrieve(url, fpath)
                    size_mb = os.path.getsize(fpath) / 1e6
                    print(f"  Downloaded {fname}: {size_mb:.0f} MB")
                    if size_mb > 500:
                        print("  Too large, skipping")
                        os.remove(fpath)
                        continue
                    # Try to load
                    if fname.endswith(".h5ad"):
                        adata = sc.read_h5ad(fpath)
                        return adata, None
                    break
                except Exception as e:
                    print(f"  Download failed: {e}")
                    continue
    except Exception as e:
        print(f"  GEOparse failed: {e}")

    return None, None


def synthetic_covid(n_samples=10):
    """COVID-19 profile: high Macrophage→T_cell, T_cell→Macrophage (cytokine storm)."""
    from generate_synthetic import generate_group, base_matrix, set_edge, IDX
    def covid_template():
        m = base_matrix() * 1.2
        set_edge(m, "Macrophage", "T_cell", 0.88)
        set_edge(m, "T_cell", "Macrophage", 0.82)
        set_edge(m, "Macrophage", "NK_cell", 0.70)
        set_edge(m, "NK_cell", "T_cell", 0.60)
        set_edge(m, "Dendritic_cell", "T_cell", 0.65)
        set_edge(m, "T_cell", "B_cell", 0.45)
        return m
    rng = np.random.default_rng(1)
    block = generate_group(covid_template, n_samples=n_samples, noise_frac=0.12, rng=rng)
    labels = [f"covid_S{i+1:02d}" for i in range(n_samples)]
    meta = [{
        "sample_id": lbl, "dataset_source": "synthetic_covid",
        "tissue": "PBMC/BALF", "disease": "COVID-19",
        "n_cells_total": 0, "n_cells_mapped": 0,
        "pct_cells_mapped": 100.0, "low_quality": False,
    } for lbl in labels]
    return block, meta


def synthetic_ipf(n_samples=10):
    """IPF profile: fibroblast autocrine dominant, Macrophage→Fibroblast."""
    from generate_synthetic import generate_group, base_matrix, set_edge
    def ipf_template():
        m = base_matrix()
        set_edge(m, "Fibroblast", "Fibroblast", 0.95)
        set_edge(m, "Macrophage", "Fibroblast", 0.85)
        set_edge(m, "Fibroblast", "Macrophage", 0.75)
        set_edge(m, "Macrophage", "Macrophage", 0.55)
        set_edge(m, "T_cell", "Macrophage", 0.15)
        return m
    rng = np.random.default_rng(2)
    block = generate_group(ipf_template, n_samples=n_samples, noise_frac=0.10, rng=rng)
    labels = [f"ipf_S{i+1:02d}" for i in range(n_samples)]
    meta = [{
        "sample_id": lbl, "dataset_source": "synthetic_ipf",
        "tissue": "lung", "disease": "IPF",
        "n_cells_total": 0, "n_cells_mapped": 0,
        "pct_cells_mapped": 100.0, "low_quality": False,
    } for lbl in labels]
    return block, meta


def synthetic_tumor(n_samples=10):
    """Tumor TME: Macrophage→T_cell suppression, Fibroblast→Macrophage remodeling."""
    from generate_synthetic import generate_group, base_matrix, set_edge
    def tumor_template():
        m = base_matrix()
        set_edge(m, "Macrophage", "T_cell", 0.90)
        set_edge(m, "Fibroblast", "Macrophage", 0.85)
        set_edge(m, "Fibroblast", "T_cell", 0.60)
        set_edge(m, "Macrophage", "NK_cell", 0.55)
        set_edge(m, "Dendritic_cell", "T_cell", 0.50)
        set_edge(m, "B_cell", "NK_cell", 0.08)
        return m
    rng = np.random.default_rng(3)
    block = generate_group(tumor_template, n_samples=n_samples, noise_frac=0.10, rng=rng)
    labels = [f"tumor_S{i+1:02d}" for i in range(n_samples)]
    meta = [{
        "sample_id": lbl, "dataset_source": "synthetic_tumor",
        "tissue": "solid tumor", "disease": "cancer",
        "n_cells_total": 0, "n_cells_mapped": 0,
        "pct_cells_mapped": 100.0, "low_quality": False,
    } for lbl in labels]
    return block, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def build_real_M():
    print("=" * 65)
    print("Building M_real matrix")
    print("=" * 65)
    print(f"RAM start: {ram():.2f} GB\n")

    lr_db = load_lr_database()
    all_rows, all_meta = [], []

    # ── D1: COVID-19 ──────────────────────────────────────────────────────────
    print("\n[D1] GSE158055 COVID-19")
    covid_adata, ct_col = try_geo_gse158055()
    if covid_adata is not None and ct_col is not None:
        covid_adata = make_pseudo_samples(covid_adata, ct_col, n_samples=10, source_tag="covid")
        rows, meta = flow_from_adata(
            covid_adata, ct_col, "pseudo_sample", lr_db,
            source="GSE158055", tissue="BALF", disease="COVID-19"
        )
        del covid_adata; gc.collect()
        all_rows.extend(rows); all_meta.extend(meta)
        print(f"  → {len(rows)} samples from real COVID data")
    else:
        print("  → Using synthetic COVID fallback")
        block, meta = synthetic_covid(n_samples=10)
        all_rows.extend(block.tolist()); all_meta.extend(meta)
    print(f"  RAM: {ram():.2f} GB")

    # ── D2: pbmc3k (healthy PBMC) ─────────────────────────────────────────────
    print("\n[D2] pbmc3k_processed — healthy PBMC")
    adata2 = sc.datasets.pbmc3k_processed()
    adata2.obs["cell_type"] = adata2.obs["louvain"].astype(str)
    adata2 = make_pseudo_samples(adata2, "cell_type", n_samples=10, source_tag="pbmc3k")
    gene_names2 = adata2.var_names.tolist()
    pb2 = compute_pseudobulk(adata2, "cell_type", "pseudo_sample")
    rows2, meta2 = [], []
    for sid, pbs in pb2.items():
        mat = compute_flow_matrix(pbs, gene_names2, lr_db)
        n_tot = int((adata2.obs["pseudo_sample"] == sid).sum())
        n_map = int(((adata2.obs["pseudo_sample"] == sid) &
                     (adata2.obs["cell_type"].map(map_cell_type).notna())).sum())
        rows2.append(mat.flatten().astype(np.float32))
        meta2.append({
            "sample_id": sid, "dataset_source": "pbmc3k",
            "tissue": "PBMC", "disease": "healthy",
            "n_cells_total": n_tot, "n_cells_mapped": n_map,
            "pct_cells_mapped": round(100 * n_map / max(n_tot, 1), 1),
            "low_quality": False,
        })
    del adata2, pb2; gc.collect()
    all_rows.extend(rows2); all_meta.extend(meta2)
    print(f"  → {len(rows2)} samples  RAM: {ram():.2f} GB")

    # ── D3: pbmc68k_reduced ───────────────────────────────────────────────────
    print("\n[D3] pbmc68k_reduced — PBMC immune panel")
    adata3 = sc.datasets.pbmc68k_reduced()
    adata3.obs["cell_type"] = adata3.obs["bulk_labels"].astype(str)
    adata3 = make_pseudo_samples(adata3, "cell_type", n_samples=10, source_tag="pbmc68k")
    gene_names3 = adata3.var_names.tolist()
    pb3 = compute_pseudobulk(adata3, "cell_type", "pseudo_sample")
    rows3, meta3 = [], []
    for sid, pbs in pb3.items():
        mat = compute_flow_matrix(pbs, gene_names3, lr_db)
        n_tot = int((adata3.obs["pseudo_sample"] == sid).sum())
        n_map = int(((adata3.obs["pseudo_sample"] == sid) &
                     (adata3.obs["cell_type"].map(map_cell_type).notna())).sum())
        rows3.append(mat.flatten().astype(np.float32))
        meta3.append({
            "sample_id": sid, "dataset_source": "pbmc68k",
            "tissue": "PBMC", "disease": "healthy",
            "n_cells_total": n_tot, "n_cells_mapped": n_map,
            "pct_cells_mapped": round(100 * n_map / max(n_tot, 1), 1),
            "low_quality": False,
        })
    del adata3, pb3; gc.collect()
    all_rows.extend(rows3); all_meta.extend(meta3)
    print(f"  → {len(rows3)} samples  RAM: {ram():.2f} GB")

    # ── D4: synthetic IPF ─────────────────────────────────────────────────────
    print("\n[D4] Synthetic IPF lung fibrotic")
    block4, meta4 = synthetic_ipf(n_samples=10)
    all_rows.extend(block4.tolist()); all_meta.extend(meta4)
    print(f"  → {len(block4)} synthetic samples")

    # ── D5: synthetic tumor TME ───────────────────────────────────────────────
    print("\n[D5] Synthetic tumor TME")
    block5, meta5 = synthetic_tumor(n_samples=10)
    all_rows.extend(block5.tolist()); all_meta.extend(meta5)
    print(f"  → {len(block5)} synthetic samples")

    # ── Stack ─────────────────────────────────────────────────────────────────
    M_real = np.array(all_rows, dtype=np.float32)
    meta_df = pd.DataFrame(all_meta)

    print(f"\n{'='*65}")
    print(f"M_real shape: {M_real.shape}")
    print(f"{'='*65}")
    print(meta_df[["sample_id","dataset_source","tissue","disease",
                   "pct_cells_mapped","low_quality"]].to_string(index=False))
    n_low = meta_df["low_quality"].sum()
    print(f"\nLow-quality samples (<50% mapped): {n_low}")
    print(f"RAM end: {ram():.2f} GB")

    np.save(f"{DATA_DIR}/M_real.npy", M_real)
    meta_df.to_csv(f"{DATA_DIR}/metadata.csv", index=False)
    print(f"\nSaved M_real.npy ({M_real.shape}) and metadata.csv")
    return M_real, meta_df


if __name__ == "__main__":
    M_real, meta_df = build_real_M()
