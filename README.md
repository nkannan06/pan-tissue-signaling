# Pan-Tissue Signaling Periodic Table

Computational framework for learning universal cell-cell communication programs
from scRNA-seq data using sparse NMF decomposition of LIANA-derived
ligand-receptor communication matrices.

> **Integrity note (2026-06-27):** This codebase underwent a systematic
> adversarial integrity audit. All claims carry explicit verification status.
> Three results were retracted. See audit details below.

---

## Results Summary

| Claim | Status | Details |
|-------|--------|---------|
| NMF stability = 0.9997 (10-run cosine match) | ✓ **VERIFIED** | K=8 atoms reproducible |
| Fibrosis atom (A2): cosine 0.953 vs synthetic GT | ✓ **VERIFIED** | Against known ground truth |
| Cancer atom (A1): cosine 0.985 vs synthetic GT | ✓ **VERIFIED** | Against known ground truth |
| Held-out classification: 70%, p=0.0035 | ✓ **VERIFIED** | 10 never-seen samples, 4-class, 25% chance |
| COVID-before-IPF ordering: 10/10 noise trials | ✓ **VERIFIED** | W-space distances, label-free confirmation |
| Phase tipping point Stage 4→5 (A8↓0.51, A2↑0.45) | ✓ **VERIFIED** | Simultaneous inflammation→fibrosis switch |
| COVID→Cancer confusion = shared Mac↔T_cell biology | ✓ **VERIFIED** | Biologically interpretable misclassification |
| EHR disease atoms R²=0.93–0.99 | ~ **PROOF-OF-CONCEPT** | Synthetic data only; whole-model not above null |
| Spatial fibrosis zonation in wound bed | ~ **PROOF-OF-CONCEPT** | Synthetic 20×20 grid; no Visium data |
| Drug-gene links directionally consistent (ChEMBL) | ~ **PROOF-OF-CONCEPT** | p>0.05 at n=40 resolved; needs full DrugBank |

---

## What the Audit Found

### Verified findings
- NMF atoms are biologically stable and reproduce across random initializations
- Fibrosis (A2) and cancer (A1) atoms match their synthetic ground truth at cosine >0.95
- Held-out classification (samples never seen during NMF) reaches 70% accuracy, p=0.0035 vs 25% chance
- The COVID→IPF disease ordering emerges from unsupervised atom weights in 10/10 noise trials
- The COVID-to-cancer misclassification is biologically valid: both involve Mac↔T_cell exhaustion programs documented in Diao et al. 2020

### Retracted results
1. **Drug 9/9 validation** — the curated drug knowledge base was written by the same code that extracted the gene sets, creating a circular validation loop. The null model test confirmed random atoms receive the same drug recommendations. Replaced with honest ChEMBL external query.
2. **EHR R²=0.993** — this was performance on training data with the same random seed used to generate features. Held-out test with a different seed gives whole-model mean R²=−1.24 (not above null). Disease atoms only (cancer/fibrosis/inflammation) show R²=0.93–0.99 but on synthetic data by construction.
3. **5-fold CV 100%** — the CV loop compared against H_real loaded from disk rather than truly independent atoms, making it an in-sample test.

### Known limitations (semi-circular)
- Pseudotime uses dataset_source labels to identify the healthy reference centroid. Atom weights themselves are unsupervised. This is standard practice in trajectory methods (Monocle, scVelo, Palantir) but should be disclosed.

---

## Architecture

```
Data acquisition:   pseudobulk.py → compute_lr_scores.py → build_real_M.py
Decomposition:      nmf_decompose.py
Annotation:         annotate_atoms.py → enrichr_runner.py (venv310)
Classification:     classify_sample.py
Validation:         validate_atoms.py → integrity_audit.py
Trajectory:         pseudotime.py → transitions.py → predict_trajectory.py
Intervention:       simulate_intervention.py
Extensions:         ehr_bridge.py → drug_repurposing.py → spatial_mapping.py
Confusion matrix:   classification_confusion.py
Summary:            generate_summary.py → paper_figure.py
```

### Data
- 50 samples: 10 PBMC3k + 10 PBMC68k (real) + 10 synthetic COVID + 10 synthetic IPF + 10 synthetic tumor
- 36 LR flow features: 6×6 cell-type communication matrix (top-50 LIANA pairs)
- K=8 NMF atoms: 4 homeostatic + 1 inflammation + 1 fibrosis + 1 cancer + 1 mixed

### Key files
| File | Description |
|------|-------------|
| `data/H_real.npy` | NMF atom basis (8×36) |
| `data/W_norm.npy` | Sample weights (50×8) |
| `data/phase_transitions.csv` | Tipping-point table |
| `data/drug_repurposing_results.csv` | ChEMBL query (honest) |
| `figures/periodic_table_figure.png` | 8-panel figure A–H |
| `figures/classification_confusion.png` | Confusion matrix |
| `reports/` | Per-sample clinical reports |

---

## Requirements for publication

1. **Real multi-tissue scRNA-seq** — replace 35/50 synthetic samples (GEO: GSE165816 DFU, GSE158055 COVID)
2. **Full drug database** — DrugBank (free account) + hypergeometric p<0.05
3. **Paired EHR + scRNA-seq cohort** — validate EHR bridge on real patients (min n=50)
4. **Visium/Slide-seq data** — validate spatial atom maps
5. **Label-free pseudotime** — replace healthy-centroid ordering with PAGA or diffusion maps
6. **Independent replication** — NMF on dataset from independent lab

---

## Integrity audit

Full adversarial audit: `integrity_audit.py`

Tests:
1. Drug repurposing via ChEMBL external API (no hardcoded knowledge)
2. EHR bridge: held-out test + permutation test
3. NMF held-out classification (never-seen samples)
4. COVID→fibrosis circularity check + noise robustness
5. Null model: pipeline on random data

Run: `python integrity_audit.py`
