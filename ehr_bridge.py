"""
Addition 1: EHR Bridge — predict atom weights from clinical variables.
Deploys the periodic table without scRNA-seq input.
"""

import sys, os, warnings, pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from annotate_atoms import _auto_label, map_atom_to_diseases
from predict_trajectory import assign_stage, apply_transition, describe_state
from simulate_intervention import apply_intervention, INTERVENTIONS

DATA_DIR   = "/Users/nitkan/pan-tissue-signaling/data"
MODELS_DIR = "/Users/nitkan/pan-tissue-signaling/models"
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ── Step 1: Build synthetic clinical dataset ──────────────────────────────────

def build_clinical_dataset(W_norm, meta):
    """
    Generate synthetic clinical variables mechanistically linked to dominant atom.
    12 features × 50 samples.
    """
    rows = []
    for i, row in meta.iterrows():
        src = row["dataset_source"]
        w   = W_norm[i]
        dom = int(np.argmax(w))   # dominant atom index (0-based)

        # Base values (healthy)
        feats = {
            "CRP":             np.random.normal(2.0,  0.8),
            "albumin":         np.random.normal(4.2,  0.2),
            "TGFb1_serum":     np.random.normal(2.5,  0.4),
            "age":             np.random.normal(42,   8.0),
            "FVC_pct":         np.random.normal(98,   6.0),
            "HbA1c":           np.random.normal(5.4,  0.3),
            "IL6_serum":       np.random.normal(3.0,  1.0),
            "neutrophil_pct":  np.random.normal(55,   6.0),
            "ferritin":        np.random.normal(120,  30),
            "LDH":             np.random.normal(180,  20),
            "CA125":           np.random.normal(12,   4.0),
            "CD8_count":       np.random.normal(450,  60),
        }

        # Modulate by source — weighted by atom loadings
        fibro_w  = w[1]   # Atom 2 = fibrosis
        inflam_w = w[7]   # Atom 8 = inflammation
        tumor_w  = w[0]   # Atom 1 = tumor

        # Fibrosis signal
        feats["CRP"]         += fibro_w * np.random.normal(43, 10)
        feats["albumin"]     -= fibro_w * np.random.normal(1.4, 0.3)
        feats["TGFb1_serum"] += fibro_w * np.random.normal(6.0, 1.2)
        feats["age"]         += fibro_w * np.random.normal(20,  5)
        feats["FVC_pct"]     -= fibro_w * np.random.normal(40,  8)
        feats["HbA1c"]       += fibro_w * np.random.normal(0.4, 0.1)

        # Inflammation signal
        feats["CRP"]            += inflam_w * np.random.normal(118, 25)
        feats["IL6_serum"]      += inflam_w * np.random.normal(42,  12)
        feats["neutrophil_pct"] += inflam_w * np.random.normal(27,  5)
        feats["ferritin"]       += inflam_w * np.random.normal(730, 200)

        # Tumor signal
        feats["LDH"]      += tumor_w * np.random.normal(200, 60)
        feats["CA125"]    += tumor_w * np.random.normal(168, 40)
        feats["CD8_count"]-= tumor_w * np.random.normal(270, 40)

        # Clip physiological bounds
        feats["albumin"]        = max(feats["albumin"], 1.0)
        feats["FVC_pct"]        = np.clip(feats["FVC_pct"], 20, 120)
        feats["neutrophil_pct"] = np.clip(feats["neutrophil_pct"], 20, 98)
        feats["CD8_count"]      = max(feats["CD8_count"], 10)
        feats["CRP"]            = max(feats["CRP"], 0.1)

        rows.append(feats)

    clinical_df = pd.DataFrame(rows)
    target_cols = [f"atom_{k+1}" for k in range(W_norm.shape[1])]
    target_df   = pd.DataFrame(W_norm, columns=target_cols)
    return pd.concat([clinical_df, target_df], axis=1), list(feats.keys()), target_cols


# ── Step 2: Train multi-output regression model ───────────────────────────────

def train_model(clinical_df, feature_cols, target_cols):
    X = clinical_df[feature_cols].values
    Y = clinical_df[target_cols].values

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.2, random_state=SEED)

    base = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=SEED)
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_tr, Y_tr)

    Y_pred = model.predict(X_te)
    r2s    = [r2_score(Y_te[:, k], Y_pred[:, k]) for k in range(Y.shape[1])]
    return model, r2s, X_te, Y_te, Y_pred


# ── Step 3: Feature importance ────────────────────────────────────────────────

def print_feature_importance(model, feature_cols, atom_labels, r2s):
    print("\n" + "="*65)
    print("EHR BRIDGE — FEATURE IMPORTANCE PER ATOM")
    print("="*65)
    for k, est in enumerate(model.estimators_):
        lbl  = atom_labels[k][:35]
        info = map_atom_to_diseases(atom_labels[k])
        imp  = est.feature_importances_
        top  = np.argsort(imp)[::-1][:3]
        top_str = ", ".join(f"{feature_cols[j]} ({imp[j]:.2f})" for j in top)
        r2   = r2s[k]
        flag = "✓" if r2 > 0.70 else "~" if r2 > 0.40 else "✗"
        print(f"  Atom {k+1:1d} ({info['category']:<12}) R²={r2:+.3f} {flag}  "
              f"| top predictors: {top_str}")
        print(f"    [{lbl}]")


# ── Step 4: Deployment simulation ────────────────────────────────────────────

def predict_from_blood_panel(model, feature_cols, atom_labels,
                              stage_centroids, deltas,
                              **kwargs) -> dict:
    """
    Predict atom weights from a blood panel.
    Call with keyword args matching feature_cols.
    Returns dict with atom weights + clinical report text.
    """
    feat_vec = np.array([kwargs.get(f, 0.0) for f in feature_cols]).reshape(1, -1)
    w_pred   = model.predict(feat_vec)[0]
    w_pred   = np.maximum(w_pred, 0)
    w_pred  /= w_pred.sum() + 1e-12

    stage = assign_stage(w_pred, stage_centroids)
    untreated = apply_transition(w_pred, deltas, stage, n_steps=2)

    dom_k   = int(np.argmax(w_pred))
    dom_lbl = atom_labels[dom_k]
    dom_info= map_atom_to_diseases(dom_lbl)

    lines = [
        "═"*55,
        "EHR-DERIVED TISSUE COMMUNICATION REPORT",
        "(No scRNA-seq required — predicted from blood panel)",
        "═"*55,
        "",
        "INPUT PANEL:",
    ]
    for f in feature_cols:
        v = kwargs.get(f)
        if v is not None:
            lines.append(f"  {f:<20} = {v}")
    lines += [
        "",
        "PREDICTED ATOM WEIGHTS:",
    ]
    order = np.argsort(w_pred)[::-1]
    for k in order[:4]:
        if w_pred[k] > 0.03:
            info = map_atom_to_diseases(atom_labels[k])
            lines.append(f"  Atom {k+1} ({info['category']:<12}): {w_pred[k]*100:.1f}%")

    lines += [
        "",
        f"DOMINANT PROGRAM: {dom_info['category'].title()} ({dom_info['diseases'][0]})",
        f"Disease stage: {stage+1}/5",
        "",
        "PREDICTED TRAJECTORY (untreated):",
        f"  +1 step: {untreated[1][dom_k]*100:.0f}% {dom_info['category']}",
        f"  +2 step: {untreated[2][dom_k]*100:.0f}% {dom_info['category']}",
        "",
        "SIMILAR DISEASE CONTEXTS:",
        "  " + ", ".join(dom_info["diseases"]),
        "═"*55,
    ]
    report = "\n".join(lines)
    return {"w_pred": w_pred, "stage": stage, "report": report}


def run_ehr_bridge():
    W_norm  = np.load(f"{DATA_DIR}/W_norm.npy")
    meta    = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")
    H_real  = np.load(f"{DATA_DIR}/H_real.npy")
    K       = H_real.shape[0]
    atom_labels     = [_auto_label(H_real[k]) for k in range(K)]
    stage_centroids = np.load(f"{DATA_DIR}/stage_centroids.npy")
    deltas          = np.load(f"{DATA_DIR}/transition_deltas.npy")

    print("Building clinical dataset...")
    clinical_df, feature_cols, target_cols = build_clinical_dataset(W_norm, meta)

    print("Training EHR bridge model...")
    model, r2s, X_te, Y_te, Y_pred = train_model(clinical_df, feature_cols, target_cols)

    print_feature_importance(model, feature_cols, atom_labels, r2s)

    # Save R² results
    r2_df = pd.DataFrame({
        "atom": [f"Atom_{k+1}" for k in range(K)],
        "label": atom_labels,
        "category": [map_atom_to_diseases(atom_labels[k])["category"] for k in range(K)],
        "r2": [round(r, 4) for r in r2s],
    })
    r2_df.to_csv(f"{DATA_DIR}/ehr_bridge_results.csv", index=False)
    print(f"\nSaved: {DATA_DIR}/ehr_bridge_results.csv")

    # Save model
    with open(f"{MODELS_DIR}/ehr_bridge_model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols,
                     "atom_labels": atom_labels}, f)
    print(f"Saved: {MODELS_DIR}/ehr_bridge_model.pkl")

    # Demo: simulated IPF patient
    print("\n" + "─"*55)
    print("DEMO: Simulated IPF patient (high TGFb1, low albumin, low FVC)")
    result = predict_from_blood_panel(
        model, feature_cols, atom_labels, stage_centroids, deltas,
        CRP=52.0, albumin=2.6, TGFb1_serum=9.8, age=68.0,
        FVC_pct=52.0, HbA1c=6.1, IL6_serum=8.0,
        neutrophil_pct=62.0, ferritin=280.0,
        LDH=195.0, CA125=15.0, CD8_count=390.0,
    )
    print(result["report"])

    return model, r2s, r2_df, clinical_df, feature_cols, target_cols, X_te, Y_te, Y_pred


if __name__ == "__main__":
    run_ehr_bridge()
