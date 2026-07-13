"""
Step 1: Download GSE154826 (breast TME) from GEO.
Prints file list inside the downloaded tar before parsing.
Falls back to synthetic data if download fails.
"""

import os
import gc
import gzip
import tarfile
import urllib.request
import GEOparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data/GSE154826"
os.makedirs(DATA_DIR, exist_ok=True)

GSE_ID = "GSE154826"


def download_gse154826():
    """Download supplementary files from GEO FTP."""
    ftp_base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154826/suppl/"
    tar_name = "GSE154826_RAW.tar"
    tar_path = os.path.join(DATA_DIR, tar_name)

    if not os.path.exists(tar_path):
        url = ftp_base + tar_name
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, tar_path)
            print(f"Downloaded to {tar_path}")
        except Exception as e:
            print(f"Direct download failed: {e}")
            return None
    else:
        print(f"Already exists: {tar_path}")

    # Print contents
    print("\n--- Files inside tar ---")
    with tarfile.open(tar_path) as tf:
        members = tf.getnames()
        for m in members:
            print(" ", m)
    print(f"Total files: {len(members)}")
    return tar_path, members


def extract_and_load(tar_path, members):
    """
    GSE154826 RAW tar contains per-sample dirs with matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz.
    We load one sample to verify, then load all and concatenate.
    """
    extract_dir = os.path.join(DATA_DIR, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir)
    print(f"\nExtracted to {extract_dir}")

    # Find sample directories (each should have matrix.mtx.gz)
    sample_dirs = {}
    for root, dirs, files in os.walk(extract_dir):
        if any(f.endswith("matrix.mtx.gz") or f == "matrix.mtx" for f in files):
            sample_id = os.path.basename(root)
            sample_dirs[sample_id] = root

    print(f"Found {len(sample_dirs)} sample directories:")
    for sid in list(sample_dirs.keys())[:5]:
        print(f"  {sid}")
    if len(sample_dirs) > 5:
        print(f"  ... and {len(sample_dirs)-5} more")

    return sample_dirs


def load_10x_sample(sample_dir, sample_id):
    """Load a 10x-format directory into AnnData."""
    import scanpy as sc
    adata = sc.read_10x_mtx(sample_dir, var_names="gene_symbols", cache=False)
    adata.obs["sample"] = sample_id
    return adata


def load_all_samples(sample_dirs, max_samples=None):
    """Load and concatenate all samples. Deletes intermediates immediately."""
    import scanpy as sc
    adatas = []
    sample_list = list(sample_dirs.items())
    if max_samples:
        sample_list = sample_list[:max_samples]

    for sid, sdir in sample_list:
        try:
            a = load_10x_sample(sdir, sid)
            # Keep only obs/var metadata + raw counts
            adatas.append(a)
            print(f"  Loaded {sid}: {a.shape}")
            del a
            gc.collect()
        except Exception as e:
            print(f"  Skipping {sid}: {e}")

    if not adatas:
        return None

    combined = ad.concat(adatas, join="outer", fill_value=0)
    del adatas
    gc.collect()
    return combined


def try_geoparse_fallback():
    """Try GEOparse to find metadata + soft file links."""
    print("\nTrying GEOparse for metadata...")
    try:
        gse = GEOparse.get_GEO(geo=GSE_ID, destdir=DATA_DIR, silent=True)
        print(f"GSE title: {gse.metadata['title'][0]}")
        print(f"Samples: {len(gse.gsms)}")
        for gsm_id, gsm in list(gse.gsms.items())[:3]:
            print(f"  {gsm_id}: {gsm.metadata.get('title', ['?'])[0]}")
        return gse
    except Exception as e:
        print(f"GEOparse failed: {e}")
        return None


if __name__ == "__main__":
    import psutil
    ram_before = psutil.virtual_memory().used / 1e9
    print(f"RAM before: {ram_before:.2f} GB\n")

    result = download_gse154826()
    if result is not None:
        tar_path, members = result
        sample_dirs = extract_and_load(tar_path, members)
        if sample_dirs:
            print("\nLoading first sample to verify format...")
            sid = list(sample_dirs.keys())[0]
            try:
                import scanpy as sc
                adata = load_10x_sample(sample_dirs[sid], sid)
                print(f"Sample shape: {adata.shape}")
                print(f"obs columns: {adata.obs.columns.tolist()}")
                print(f"First 5 genes: {adata.var_names[:5].tolist()}")
                del adata
                gc.collect()
            except Exception as e:
                print(f"Sample load failed: {e}")
    else:
        try_geoparse_fallback()

    ram_after = psutil.virtual_memory().used / 1e9
    print(f"\nRAM after: {ram_after:.2f} GB")
