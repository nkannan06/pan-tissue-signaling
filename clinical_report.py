"""
Task B: Clinical report generator.
Produces a structured one-page summary for a given sample.
"""

import sys, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from annotate_atoms import _auto_label, map_atom_to_diseases, DISEASE_MAP
from predict_trajectory import assign_stage, apply_transition, describe_state
from simulate_intervention import apply_intervention, INTERVENTIONS

DATA_DIR    = "/Users/nitkan/pan-tissue-signaling/data"
REPORTS_DIR = "/Users/nitkan/pan-tissue-signaling/reports"
N_STAGES    = 5

# Hardcoded top LR pairs per dominant edge (from annotate_atoms run)
TOP_LR = {
    "fibrosis":     [
        ("GRN", "TNFRSF1B",  "Macrophage",  "Fibroblast"),
        ("TNFSF13B", "HLA-DPB1", "Macrophage", "Fibroblast"),
        ("POSTN", "ITGAV",   "Fibroblast",  "Fibroblast"),
    ],
    "cancer":       [
        ("IL15", "IL15RA",   "Macrophage",  "T_cell"),
        ("CD274", "PDCD1",   "Macrophage",  "T_cell"),
        ("VEGFA", "FLT1",    "Fibroblast",  "Macrophage"),
    ],
    "inflammation": [
        ("IL6", "IL6R",      "T_cell",      "Macrophage"),
        ("TNF", "TNFRSF1A",  "Macrophage",  "T_cell"),
        ("CXCL10","CXCR3",   "Macrophage",  "T_cell"),
    ],
    "homeostasis":  [
        ("LTB", "LTBR",      "NK_cell",     "NK_cell"),
        ("KLRB1","CLEC2D",   "NK_cell",     "Macrophage"),
        ("CD40L","CD40",     "Dendritic_cell","B_cell"),
    ],
}

RISK_LABELS = {1:"LOW", 2:"LOW-MODERATE", 3:"MODERATE", 4:"HIGH", 5:"HIGH"}


def _intervention_label(name: str) -> str:
    return name.split("\n")[0]


def intervention_benefit(untreated_final_w, treated_final_w):
    """Rate the benefit of an intervention based on fibrosis atom change."""
    delta = untreated_final_w[1] - treated_final_w[1]   # atom 2 = index 1 = fibrosis
    if delta > 0.15:
        return "substantial reduction in dominant program"
    elif delta > 0.05:
        return "moderate reduction, partial diversion"
    elif delta > 0.01:
        return "minimal reduction"
    else:
        return "minimal — target atom is secondary driver"


def generate_report(sample_id: str) -> str:
    W_norm          = np.load(f"{DATA_DIR}/W_norm.npy")
    meta_pt         = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")
    stage_centroids = np.load(f"{DATA_DIR}/stage_centroids.npy")
    deltas          = np.load(f"{DATA_DIR}/transition_deltas.npy")
    H_real          = np.load(f"{DATA_DIR}/H_real.npy")
    K               = H_real.shape[0]
    atom_labels     = [_auto_label(H_real[k]) for k in range(K)]

    # Find sample
    row = meta_pt[meta_pt["sample_id"] == sample_id]
    if len(row) == 0:
        raise ValueError(f"Sample '{sample_id}' not found in metadata.")
    row     = row.iloc[0]
    idx     = row.name
    w       = W_norm[idx]
    stage   = assign_stage(w, stage_centroids)
    src     = row["dataset_source"]
    prog    = row["progression_score"]

    # Infer tissue / disease from source
    SRC_META = {
        "pbmc3k":          ("Peripheral blood", "Healthy"),
        "pbmc68k":         ("Peripheral blood", "Healthy"),
        "synthetic_covid": ("Lung / blood",     "COVID-19 severe"),
        "synthetic_ipf":   ("Lung",             "IPF / Pulmonary fibrosis"),
        "synthetic_tumor": ("Lung / solid tissue","Tumor microenvironment"),
    }
    tissue, disease = SRC_META.get(src, ("Unknown", "Unknown"))

    # Primary / secondary atom
    order = np.argsort(w)[::-1]
    primary_k   = order[0];  primary_w   = w[primary_k]
    secondary_k = order[1];  secondary_w = w[secondary_k]
    primary_lbl   = atom_labels[primary_k]
    secondary_lbl = atom_labels[secondary_k]
    primary_info  = map_atom_to_diseases(primary_lbl)
    secondary_info= map_atom_to_diseases(secondary_lbl)
    primary_cat   = primary_info["category"].title()
    secondary_cat = secondary_info["category"].title()

    # Untreated trajectory
    untreated = apply_transition(w, deltas, stage, n_steps=2)

    # Attractor check: < 3% change in dominant atom across 2 steps
    dom_now = untreated[0][primary_k]
    dom_t2  = untreated[2][primary_k]
    is_locked = abs(dom_now - dom_t2) < 0.03
    risk_lbl = RISK_LABELS.get(stage + 1, "UNKNOWN")
    if is_locked:
        risk_detail = f"{risk_lbl} — sample is in stable {primary_info['category']} attractor"
    else:
        risk_detail = f"{risk_lbl} — progressing toward {primary_info['category']}"

    # Intervention trajectories
    intv_results = {}
    for name, atom_scales in INTERVENTIONS.items():
        w_tx1 = apply_intervention(untreated[1], atom_scales)
        traj_tx = apply_transition(w_tx1, deltas, min(stage+1, N_STAGES-2), n_steps=1)
        w_tx2 = traj_tx[-1]
        intv_results[name] = (w_tx1, w_tx2)

    # Best recommendation: which intervention reduces primary atom most at step 2
    best_name, best_reduction = None, 0.0
    for name, (w_tx1, w_tx2) in intv_results.items():
        reduction = untreated[2][primary_k] - w_tx2[primary_k]
        if reduction > best_reduction:
            best_reduction, best_name = reduction, name

    # Top LR pairs for primary category
    lr_pairs = TOP_LR.get(primary_info["category"], TOP_LR["homeostasis"])

    # Similar disease contexts
    all_diseases = primary_info["diseases"]

    # ── Build report string ───────────────────────────────────────────────────
    W = "═" * 55
    lines = [
        W,
        "TISSUE COMMUNICATION ANALYSIS REPORT",
        f"Sample: {sample_id} | Tissue: {tissue} | Disease: {disease}",
        W,
        "",
        "CURRENT STATE",
        f"Primary program:   {primary_cat} (Atom {primary_k+1}) — {primary_w*100:.0f}% loading",
        f"Secondary program: {secondary_cat} (Atom {secondary_k+1}) — {secondary_w*100:.0f}% loading",
        f"Disease progression stage: {stage+1}/{N_STAGES} ({'late' if stage>=3 else 'mid' if stage>=2 else 'early'})",
        f"Progression score: {prog:.3f}",
        "",
        "TRAJECTORY (untreated)",
        f"Step +1: {untreated[1][primary_k]*100:.0f}% {primary_info['category']} (A{primary_k+1})"
        + (f" + {untreated[1][order[1]]*100:.0f}% {map_atom_to_diseases(atom_labels[order[1]])['category']} (A{order[1]+1})" if untreated[1][order[1]] > 0.05 else ""),
        f"Step +2: {untreated[2][primary_k]*100:.0f}% {primary_info['category']} (A{primary_k+1})"
        + (" — stable attractor" if is_locked else " — progressing"),
        f"Risk: {risk_detail}",
        "",
        "INTERVENTION MODELING",
    ]

    for name, (w_tx1, w_tx2) in intv_results.items():
        clean = _intervention_label(name)
        benefit = intervention_benefit(untreated[2], w_tx2)
        lines += [
            f"{clean}:",
            f"  → Step +1: {w_tx1[primary_k]*100:.0f}% {primary_info['category']}"
            f" | Step +2: {w_tx2[primary_k]*100:.0f}% {primary_info['category']}",
            f"  → Benefit: {benefit}",
        ]

    if best_name:
        clean_best = _intervention_label(best_name)
        lines += [
            f"RECOMMENDATION: {clean_best} at this stage",
        ]
    else:
        lines += ["RECOMMENDATION: monitoring only — homeostatic state"]

    lines += [
        "",
        "KEY LIGAND-RECEPTOR DRIVERS",
        f"Top 3 active LR pairs in dominant atom ({primary_cat}):",
    ]
    for i, (lig, rec, sender, receiver) in enumerate(lr_pairs, 1):
        lines.append(f"  {i}. {lig}:{rec} ({sender}→{receiver})")

    diseases_str = ", ".join(all_diseases)
    lines += [
        "",
        "SIMILAR DISEASE CONTEXTS",
        f"This communication profile matches: {diseases_str}",
        W,
    ]

    report = "\n".join(lines)
    return report


def run_reports():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    meta_pt = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")

    # Pick 3 test samples
    def pick_sample(src, rank="mid"):
        sub = meta_pt[meta_pt["dataset_source"] == src].sort_values("progression_score")
        n = len(sub)
        if rank == "mid":
            return sub.iloc[n // 2]["sample_id"]
        elif rank == "late":
            return sub.iloc[-1]["sample_id"]
        else:
            return sub.iloc[0]["sample_id"]

    test_samples = [
        pick_sample("synthetic_ipf", "late"),
        pick_sample("synthetic_covid", "mid"),
        pick_sample("pbmc3k", "early"),
    ]

    for sid in test_samples:
        print(f"\nGenerating report for: {sid}")
        report = generate_report(sid)
        print(report)
        path = f"{REPORTS_DIR}/{sid}_report.txt"
        with open(path, "w") as f:
            f.write(report + "\n")
        print(f"\n→ Saved: {path}")


if __name__ == "__main__":
    run_reports()
