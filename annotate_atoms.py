"""
Step 7: Biological annotation of each NMF atom.
7a: Top LR pairs per high-weight atom edge (reconstructed from real expression)
7b: Pathway enrichment via gseapy in venv310 (subprocess), manual fallback
7c: Clinical disease mapping
"""

import sys, gc, warnings, os, json, subprocess, tempfile
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from liana.resource import select_resource
from pseudobulk import compute_pseudobulk, UNIVERSAL_TYPES, map_cell_type

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
N_TYPES  = len(UNIVERSAL_TYPES)
IDX      = {t: i for i, t in enumerate(UNIVERSAL_TYPES)}


# ── 7a helpers ────────────────────────────────────────────────────────────────

def build_mean_expr_from_real():
    """
    Reconstruct mean pseudobulk expression across real PBMC datasets.
    Returns {cell_type: mean_expr_array}, gene_names.
    """
    print("Loading pbmc3k for expression reconstruction...")
    adata = sc.datasets.pbmc3k_processed()
    adata.obs["cell_type"] = adata.obs["louvain"].astype(str)
    adata.obs["sample"] = "all"

    gene_names = adata.var_names.tolist()
    pb = compute_pseudobulk(adata, "cell_type", "sample")
    expr = pb["all"]   # {cell_type: array}
    del adata; gc.collect()

    # also pbmc68k for Dendritic richness
    print("Loading pbmc68k for DC expression...")
    adata2 = sc.datasets.pbmc68k_reduced()
    adata2.obs["cell_type"] = adata2.obs["bulk_labels"].astype(str)
    adata2.obs["sample"] = "all"
    gene_names2 = adata2.var_names.tolist()
    pb2 = compute_pseudobulk(adata2, "cell_type", "sample")
    expr2 = pb2["all"]
    del adata2; gc.collect()

    return expr, gene_names, expr2, gene_names2


def top_lr_for_edge(sender, receiver, expr, gene_names, lr_db, n=5):
    """Return top-n LR pairs by product score for a given sender→receiver edge."""
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    L_expr = expr.get(sender, np.zeros(len(gene_names)))
    R_expr = expr.get(receiver, np.zeros(len(gene_names)))

    lr_present = lr_db[
        lr_db["ligand"].isin(gene_idx) & lr_db["receptor"].isin(gene_idx)
    ].copy()

    if len(lr_present) == 0:
        return []

    l_vals = L_expr[[gene_idx[g] for g in lr_present["ligand"]]]
    r_vals = R_expr[[gene_idx[g] for g in lr_present["receptor"]]]
    scores = l_vals * r_vals

    lr_present = lr_present.copy()
    lr_present["score"] = scores
    top = lr_present.nlargest(n, "score")
    return [(row.ligand, row.receptor, row.score) for _, row in top.iterrows()]


def annotate_atom_lr(H, lr_db, expr, gene_names, atom_idx, edge_thresh_pct=0.5):
    """
    For atom k, find edges with weight > thresh * max_weight, return top LR pairs.
    """
    mat = H[atom_idx].reshape(N_TYPES, N_TYPES)
    thresh = mat.max() * edge_thresh_pct
    results = {}
    for i, sender in enumerate(UNIVERSAL_TYPES):
        for j, receiver in enumerate(UNIVERSAL_TYPES):
            if mat[i, j] >= thresh and mat[i, j] > 0:
                lrs = top_lr_for_edge(sender, receiver, expr, gene_names, lr_db, n=5)
                if lrs:
                    results[f"{sender}→{receiver}"] = lrs
    return results, mat


# ── 7b: Pathway enrichment (manual) ─────────────────────────────────────────

# Curated ligand→pathway mapping for top immune/stroma signaling genes
LIGAND_PATHWAY = {
    # Inflammation / Cytokine storm
    "LGALS9":  ["Inflammatory Response", "Immune Checkpoint"],
    "MIF":     ["Inflammatory Response", "Cytokine Signaling"],
    "IL6":     ["IL6-JAK-STAT3 Signaling", "Inflammatory Response"],
    "TNF":     ["TNF Signaling", "Inflammatory Response", "Apoptosis"],
    "IL1B":    ["IL2-STAT5 Signaling", "Inflammatory Response"],
    "CXCL10":  ["Interferon Gamma Response", "Inflammatory Response"],
    "CCL2":    ["Inflammatory Response", "IL6-JAK-STAT3 Signaling"],
    "CCL5":    ["Inflammatory Response", "Interferon Gamma Response"],
    "IFNG":    ["Interferon Gamma Response", "Immune Checkpoint"],
    # Fibrosis / ECM remodeling
    "POSTN":   ["Epithelial Mesenchymal Transition", "TGF-Beta Signaling"],
    "FN1":     ["Epithelial Mesenchymal Transition", "Coagulation"],
    "COL1A1":  ["Epithelial Mesenchymal Transition", "TGF-Beta Signaling"],
    "TGFB1":   ["TGF-Beta Signaling", "Epithelial Mesenchymal Transition"],
    "PDGFB":   ["PDGF Signaling", "Angiogenesis"],
    "MMP1":    ["Epithelial Mesenchymal Transition", "Coagulation"],
    "MMP9":    ["Epithelial Mesenchymal Transition", "Angiogenesis"],
    "THBS1":   ["Coagulation", "Angiogenesis", "TGF-Beta Signaling"],
    # Tumor / immune evasion
    "SPP1":    ["Epithelial Mesenchymal Transition", "Inflammatory Response"],
    "VEGFA":   ["Angiogenesis", "Hypoxia"],
    "CXCL12":  ["Angiogenesis", "Inflammatory Response"],
    "CD274":   ["Immune Checkpoint", "Interferon Gamma Response"],  # PD-L1
    "LGALS1":  ["Immune Checkpoint", "Apoptosis"],
    "HLA-A":   ["Allograft Rejection", "Interferon Gamma Response"],
    # NK / T cell programs
    "GZMA":    ["Apoptosis", "Inflammatory Response"],
    "PRF1":    ["Apoptosis"],
    "GNLY":    ["Apoptosis", "Inflammatory Response"],
    "FASLG":   ["Apoptosis", "Immune Checkpoint"],
    "KLRB1":   ["NK-mediated Cytotoxicity"],
    # DC / antigen presentation
    "HLA-DRA": ["Allograft Rejection", "Interferon Gamma Response"],
    "HLA-DRB1":["Allograft Rejection", "Interferon Gamma Response"],
    "CD86":    ["Allograft Rejection", "T Cell Activation"],
    "CCL22":   ["Inflammatory Response", "T Cell Activation"],
    "CCL17":   ["Inflammatory Response"],
}

def pathway_enrichment_manual(ligands):
    """Tally pathways from ligand list. Returns sorted (pathway, count, ligand_hits) list."""
    from collections import Counter
    pathway_counts = Counter()
    pathway_genes  = {}
    for lig in ligands:
        for pw in LIGAND_PATHWAY.get(lig, []):
            pathway_counts[pw] += 1
            pathway_genes.setdefault(pw, []).append(lig)
    ranked = [(pw, cnt, pathway_genes[pw]) for pw, cnt in pathway_counts.most_common()]
    return ranked


# ── 7c: Clinical disease mapping ─────────────────────────────────────────────

DISEASE_MAP = {
    "Macrophage→T_cell / Fibroblast→Macrophage": {
        "diseases": ["Breast cancer TME", "NSCLC", "Pancreatic cancer", "Melanoma"],
        "category": "cancer", "strength": 1.0,
    },
    "Fibroblast→Fibroblast / Macrophage→Fibroblast": {
        "diseases": ["IPF", "Liver cirrhosis", "DFU non-healing", "Keloid", "Systemic sclerosis"],
        "category": "fibrosis", "strength": 1.0,
    },
    "T_cell→Macrophage / Macrophage→T_cell": {
        "diseases": ["COVID-19 severe", "Sepsis", "IBD flare", "DFU inflammation"],
        "category": "inflammation", "strength": 1.0,
    },
    "T_cell→Macrophage / Macrophage→Macrophage": {
        "diseases": ["COVID-19", "Sepsis", "Rheumatoid arthritis"],
        "category": "inflammation", "strength": 0.8,
    },
    "NK_cell→NK_cell / Macrophage→NK_cell": {
        "diseases": ["Viral infection", "NK-cell lymphoma", "Healthy surveillance"],
        "category": "homeostasis", "strength": 0.6,
    },
    "Dendritic_cell→B_cell / Dendritic_cell→NK_cell": {
        "diseases": ["Vaccine response", "Allergy", "SLE"],
        "category": "homeostasis", "strength": 0.6,
    },
    "Dendritic_cell→Dendritic_cell / Dendritic_cell→Macrophage": {
        "diseases": ["Antigen presentation (homeostatic)", "Sarcoidosis"],
        "category": "homeostasis", "strength": 0.5,
    },
    "Macrophage→Dendritic_cell / B_cell→Dendritic_cell": {
        "diseases": ["Autoimmune priming", "Lupus", "Sjögren's"],
        "category": "homeostasis", "strength": 0.5,
    },
    "T_cell→NK_cell / Macrophage→Macrophage": {
        "diseases": ["Viral clearance", "CAR-T bystander effects"],
        "category": "homeostasis", "strength": 0.5,
    },
}

def map_atom_to_diseases(atom_label):
    """Find best disease map entry for an atom label string."""
    # Exact match first
    if atom_label in DISEASE_MAP:
        return DISEASE_MAP[atom_label]
    # Partial match — find entry where both top edges appear in label
    best, best_overlap = None, 0
    for key, val in DISEASE_MAP.items():
        key_words = set(key.replace("/","").replace("→"," ").split())
        label_words = set(atom_label.replace("/","").replace("→"," ").split())
        overlap = len(key_words & label_words)
        if overlap > best_overlap:
            best, best_overlap = val, overlap
    return best or {"diseases": ["Unknown / novel program"], "category": "unknown", "strength": 0.3}


# ── gseapy via subprocess (venv310) ──────────────────────────────────────────

VENV310_PYTHON = "/Users/nitkan/pan-tissue-signaling/venv310/bin/python3"
ENRICHR_SCRIPT = "/Users/nitkan/pan-tissue-signaling/enrichr_runner.py"

def run_gseapy_enrichr(atom_gene_lists: list) -> list:
    """
    Call enrichr_runner.py in venv310 via subprocess.
    atom_gene_lists: [{atom: int, genes: [str]}, ...]
    Returns list of result dicts.
    """
    payload = json.dumps(atom_gene_lists)
    try:
        proc = subprocess.run(
            [VENV310_PYTHON, ENRICHR_SCRIPT],
            input=payload, capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0:
            print(f"  [gseapy subprocess error]: {proc.stderr[:300]}")
            return []
        if proc.stderr:
            print(proc.stderr.strip())
        return json.loads(proc.stdout)
    except Exception as e:
        print(f"  [gseapy subprocess failed]: {e}")
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_annotation():
    H_real  = np.load(f"{DATA_DIR}/H_real.npy")
    K       = H_real.shape[0]
    lr_db   = select_resource("consensus")
    print(f"LIANA DB: {len(lr_db)} LR pairs\n")

    # Build mean expression from real data
    expr1, genes1, expr2, genes2 = build_mean_expr_from_real()
    all_genes = genes1
    expr_merged = {ct: expr1.get(ct, np.zeros(len(genes1))) for ct in UNIVERSAL_TYPES}

    # ── Collect ligands per atom for batch enrichr call ───────────────────────
    atom_ligand_lists = []
    per_atom_data = {}

    for k in range(K):
        atom_label = _auto_label(H_real[k])
        lr_results, mat = annotate_atom_lr(H_real, lr_db, expr_merged, all_genes,
                                           atom_idx=k, edge_thresh_pct=0.45)
        all_ligands = list({lig for edge_lrs in lr_results.values()
                            for lig, rec, score in edge_lrs if score > 0})
        per_atom_data[k] = {
            "label": atom_label,
            "lr_results": lr_results,
            "ligands": all_ligands,
        }
        if all_ligands:
            atom_ligand_lists.append({"atom": k, "genes": all_ligands})

    # ── Run enrichr in batch via venv310 ─────────────────────────────────────
    print("Running Enrichr (gseapy in venv310)...")
    enrichr_results = {}
    if atom_ligand_lists:
        raw = run_gseapy_enrichr(atom_ligand_lists)
        for res in raw:
            enrichr_results[res["atom"]] = res
    print(f"  Enrichr returned results for {len(enrichr_results)} atoms\n")

    # ── Build annotations and CSV rows ────────────────────────────────────────
    annotations = {}
    csv_rows    = []

    for k in range(K):
        d        = per_atom_data[k]
        atom_label  = d["label"]
        lr_results  = d["lr_results"]
        ligands     = d["ligands"]
        disease_info = map_atom_to_diseases(atom_label)

        # Pathway enrichment: gseapy if available, else manual
        enrichr_res = enrichr_results.get(k, {})
        if enrichr_res.get("pathways"):
            pathways_display = [
                (p["Term"], p["Adjusted P-value"],
                 p.get("Overlap","?"), p.get("Genes","")[:60])
                for p in enrichr_res["pathways"]
            ]
            pathway_source = enrichr_res.get("gene_sets", "Enrichr")
            pathway_str    = "; ".join(
                f"{t} (adjp={pv:.3g})" for t, pv, _, _ in pathways_display[:3]
            )
        else:
            manual = pathway_enrichment_manual(ligands)
            pathways_display = [(pw, None, None, str(genes)) for pw, cnt, genes in manual[:3]]
            pathway_source = "manual"
            pathway_str    = "; ".join(t for t, *_ in pathways_display[:3]) or "none"

        annotations[k] = {
            "label": atom_label,
            "lr_pairs": lr_results,
            "pathways": pathways_display,
            "pathway_source": pathway_source,
            "diseases": disease_info,
        }

        # Print
        print(f"{'='*65}")
        print(f"Atom {k+1}: {atom_label}")
        print(f"  Disease category : {disease_info['category']}  (strength={disease_info['strength']})")
        print(f"  Diseases         : {', '.join(disease_info['diseases'][:3])}")
        if lr_results:
            print(f"  Top LR pairs:")
            for edge, lrs in lr_results.items():
                top_str = ", ".join(f"{l}:{r}({s:.3f})" for l, r, s in lrs[:3])
                print(f"    {edge}: {top_str}")
        print(f"  Pathways [{pathway_source}]: {pathway_str}")

        # CSV row per atom
        for rank, (term, pval, overlap, genes) in enumerate(pathways_display[:3]):
            csv_rows.append({
                "atom": k + 1,
                "atom_label": atom_label,
                "disease_category": disease_info["category"],
                "pathway_rank": rank + 1,
                "pathway_term": term,
                "adj_pvalue": pval,
                "overlap": overlap,
                "source": pathway_source,
                "top_ligands": "; ".join(ligands[:10]),
                "diseases": "; ".join(disease_info["diseases"][:3]),
            })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = f"{DATA_DIR}/atom_pathway_annotations.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\n{'='*65}")
    print(f"Saved: {csv_path}  ({len(csv_rows)} rows)")
    return annotations


def _auto_label(atom_vec, top_n=2):
    from itertools import product as iproduct
    mat = atom_vec.reshape(N_TYPES, N_TYPES)
    pairs = [(mat[i, j], UNIVERSAL_TYPES[i], UNIVERSAL_TYPES[j])
             for i, j in iproduct(range(N_TYPES), repeat=2)]
    pairs.sort(reverse=True)
    return " / ".join(f"{s}→{r}" for _, s, r in pairs[:top_n])


if __name__ == "__main__":
    ann = run_annotation()
    # Save for paper figure
    import json, pickle
    with open(f"{DATA_DIR}/annotations.pkl", "wb") as f:
        pickle.dump(ann, f)
    print(f"\nSaved annotations.pkl")
