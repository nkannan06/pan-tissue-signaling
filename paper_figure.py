"""
Step 9: Publication-quality 4-panel figure.
Panel A: 2×4 grid of atom heatmaps
Panel B: W matrix (sample × atom loadings)
Panel C: cosine similarity real vs synthetic atoms
Panel D: atom-disease bubble chart
"""

import sys, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from itertools import product as iproduct

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nitkan/pan-tissue-signaling")

from annotate_atoms import _auto_label, map_atom_to_diseases, DISEASE_MAP
from nmf_decompose import row_normalize, run_nmf

DATA_DIR = "/Users/nitkan/pan-tissue-signaling/data"
FIG_DIR  = "/Users/nitkan/pan-tissue-signaling/figures"
os.makedirs(FIG_DIR, exist_ok=True)

UNIVERSAL_TYPES = ["Macrophage", "T_cell", "B_cell", "NK_cell", "Dendritic_cell", "Fibroblast"]
SHORT = ["Mac", "Tcell", "Bcell", "NK", "DC", "Fibro"]
N     = len(UNIVERSAL_TYPES)


def cosine_sim(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def make_figure():
    # ── Load data ─────────────────────────────────────────────────────────────
    H_real  = np.load(f"{DATA_DIR}/H_real.npy")
    H_synth = np.load(f"{DATA_DIR}/H.npy")
    W_real  = np.load(f"{DATA_DIR}/W_real.npy")
    meta    = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    M_real  = np.load(f"{DATA_DIR}/M_real.npy")

    try:
        with open(f"{DATA_DIR}/annotations.pkl", "rb") as f:
            ann = pickle.load(f)
    except FileNotFoundError:
        ann = {}

    K       = H_real.shape[0]
    K_synth = H_synth.shape[0]
    labels  = meta["sample_id"].tolist()
    sources = meta["dataset_source"].tolist()
    atom_labels = [_auto_label(H_real[k]) for k in range(K)]

    W_norm = W_real / (W_real.sum(axis=1, keepdims=True) + 1e-12)
    sim    = cosine_sim(H_real, H_synth)

    # ── Load trajectory data (for Panel E) ───────────────────────────────────
    try:
        meta_pt    = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")
        embedding  = np.load(f"{DATA_DIR}/umap_embedding.npy")
        traj_ok    = True
    except FileNotFoundError:
        traj_ok = False

    # ── Load EHR bridge data (Panel F) ───────────────────────────────────────
    try:
        import pickle as _pkl
        with open(f"{DATA_DIR.replace('data','models')}/ehr_bridge_model.pkl","rb") as _f:
            _ehr = _pkl.load(_f)
        ehr_r2 = pd.read_csv(f"{DATA_DIR}/ehr_bridge_results.csv")
        from ehr_bridge import build_clinical_dataset, train_model
        _W_n  = np.load(f"{DATA_DIR}/W_norm.npy")
        _meta = pd.read_csv(f"{DATA_DIR}/meta_with_pseudotime.csv")
        _cdf, _fcols, _tcols = build_clinical_dataset(_W_n, _meta)
        from sklearn.model_selection import train_test_split
        _Xtr, _Xte, _Ytr, _Yte = train_test_split(
            _cdf[_fcols].values, _cdf[_tcols].values, test_size=0.2, random_state=42)
        _Ypred = _ehr["model"].predict(_Xte)
        ehr_ok = True
    except Exception:
        ehr_ok = False

    # ── Load drug repurposing data (Panel G) ─────────────────────────────────
    try:
        drug_df = pd.read_csv(f"{DATA_DIR}/drug_repurposing_results.csv")
        drug_ok = True
    except FileNotFoundError:
        drug_ok = False

    # ── Load spatial data (Panel H) ───────────────────────────────────────────
    spatial_img_path = f"{FIG_DIR}/spatial_healer_vs_nonhealer.png"
    spatial_ok = os.path.exists(spatial_img_path)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 48))
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(6, 2, figure=fig,
                               hspace=0.42, wspace=0.32,
                               left=0.06, right=0.97,
                               top=0.97, bottom=0.01,
                               height_ratios=[1, 1, 0.55, 0.65, 0.65, 0.70])

    # ── Panel A: 2×4 atom heatmaps ───────────────────────────────────────────
    ax_A_title = fig.add_subplot(outer[0, 0])
    ax_A_title.set_visible(False)
    inner_A = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=outer[0, 0], hspace=0.55, wspace=0.45
    )

    cmap_atom = "YlOrRd"
    for k in range(K):
        r, c = divmod(k, 4)
        ax = fig.add_subplot(inner_A[r, c])
        mat = H_real[k].reshape(N, N)
        disease_info = map_atom_to_diseases(atom_labels[k])
        cat = disease_info["category"]
        cat_colors = {"cancer":"#d62728","fibrosis":"#8c564b",
                      "inflammation":"#ff7f0e","homeostasis":"#1f77b4","unknown":"#7f7f7f"}

        im = ax.imshow(mat, cmap=cmap_atom, aspect="auto",
                       vmin=0, vmax=H_real.max() * 0.9)
        ax.set_xticks(range(N)); ax.set_xticklabels(SHORT, fontsize=5, rotation=45, ha="right")
        ax.set_yticks(range(N)); ax.set_yticklabels(SHORT, fontsize=5)
        # value annotations
        for i, j in iproduct(range(N), repeat=2):
            v = mat[i, j]
            if v > mat.max() * 0.3:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=4.5, color="white" if v > mat.max()*0.6 else "black")
        short_lbl = atom_labels[k][:28]
        ax.set_title(f"Atom {k+1}\n{short_lbl}", fontsize=6, pad=3,
                     color=cat_colors.get(cat, "#333"))
        ax.set_xlabel("Receiver", fontsize=5)
        ax.set_ylabel("Sender",   fontsize=5)
        # colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(cat_colors.get(cat, "#333"))
            spine.set_linewidth(1.5)

    # y-position helper for panel labels (6-row grid)
    _row_tops = [0.975, 0.975, 0.655, 0.48, 0.345, 0.185]
    def _y_label(row): return _row_tops[row]

    # Panel A label
    fig.text(0.06, _y_label(0), "A", fontsize=18, fontweight="bold", va="top")
    fig.text(0.08, _y_label(0),
             "Universal Cell–Cell Communication Programs\n(Pan-Tissue Signaling Periodic Table)",
             fontsize=9, va="top", color="#333")

    # ── Panel B: W matrix heatmap ─────────────────────────────────────────────
    ax_B = fig.add_subplot(outer[0, 1])
    fig.text(0.535, _y_label(0), "B", fontsize=18, fontweight="bold", va="top")
    fig.text(0.555, _y_label(0), "Sample–Atom Loading Matrix", fontsize=9, va="top", color="#333")

    src_order  = ["synthetic_covid","pbmc3k","pbmc68k","synthetic_ipf","synthetic_tumor"]
    src_labels = ["COVID-19","Healthy PBMC (3k)","Healthy PBMC (68k)","IPF fibrotic","Tumor TME"]
    src_colors = {"synthetic_covid":"#2ca02c","pbmc3k":"#ff7f0e","pbmc68k":"#9467bd",
                  "synthetic_ipf":"#8c564b","synthetic_tumor":"#d62728"}
    cat_patch = [mpatches.Patch(color=src_colors[s], label=l)
                 for s, l in zip(src_order, src_labels)]

    df_W = pd.DataFrame(W_norm, index=labels,
                        columns=[f"A{k+1}" for k in range(K)])
    sns.heatmap(df_W, ax=ax_B, cmap="Blues", linewidths=0.1, linecolor="#eee",
                cbar_kws={"label": "Normalized loading", "shrink": 0.5},
                vmin=0, vmax=W_norm.max(),
                xticklabels=True, yticklabels=False)

    # Group separators & labels
    n_per = 10
    for gi, src in enumerate(src_order):
        y_mid = gi * n_per + n_per / 2
        if gi > 0:
            ax_B.axhline(gi * n_per, color="white", lw=1.5)
        ax_B.text(-0.3, y_mid, src_labels[gi], ha="right", va="center",
                  fontsize=7, color=src_colors[src], fontweight="bold",
                  transform=ax_B.get_yaxis_transform())

    ax_B.set_xlabel("Atom", fontsize=8)
    ax_B.set_ylabel("")
    ax_B.set_title("", pad=0)
    ax_B.tick_params(axis="x", labelsize=7)

    # ── Panel C: cosine similarity heatmap ───────────────────────────────────
    ax_C = fig.add_subplot(outer[1, 0])
    fig.text(0.06, _y_label(1), "C", fontsize=18, fontweight="bold", va="top")
    fig.text(0.08, _y_label(1), "Real vs Synthetic Atom Correspondence", fontsize=9, va="top", color="#333")

    synth_labels_short = [_auto_label(H_synth[k], top_n=1) for k in range(K_synth)]
    real_labels_short  = [f"R{k+1}: {_auto_label(H_real[k], top_n=1)}" for k in range(K)]

    sim_df = pd.DataFrame(sim,
                          index=real_labels_short,
                          columns=[f"S{i+1}:\n{l[:20]}" for i, l in enumerate(synth_labels_short)])
    mask = sim < 0.3   # grey out low-similarity cells
    sns.heatmap(sim_df, ax=ax_C, cmap="RdYlGn", vmin=0, vmax=1,
                annot=True, fmt=".2f", linewidths=0.4,
                annot_kws={"size": 7},
                cbar_kws={"label": "Cosine similarity", "shrink": 0.6})

    ax_C.set_title("", pad=0)
    ax_C.set_xlabel("Synthetic atoms (ground truth)", fontsize=8)
    ax_C.set_ylabel("Real data atoms", fontsize=8)
    ax_C.tick_params(axis="both", labelsize=7)

    # ── Panel D: bubble chart atom × disease ─────────────────────────────────
    ax_D = fig.add_subplot(outer[1, 1])
    fig.text(0.535, _y_label(1), "D", fontsize=18, fontweight="bold", va="top")
    fig.text(0.555, _y_label(1), "Atom–Disease Clinical Map", fontsize=9, va="top", color="#333")

    # Build disease list and atom associations
    disease_rows = []
    cat_col = {"cancer":"#d62728","fibrosis":"#8c564b",
               "inflammation":"#ff7f0e","homeostasis":"#1f77b4","unknown":"#aaa"}

    for k in range(K):
        lbl = atom_labels[k]
        d   = map_atom_to_diseases(lbl)
        for disease in d["diseases"][:3]:
            disease_rows.append({
                "atom": k + 1,
                "disease": disease,
                "strength": d["strength"],
                "category": d["category"],
            })

    bubble_df = pd.DataFrame(disease_rows).drop_duplicates(subset=["atom","disease"])
    all_diseases = bubble_df["disease"].unique().tolist()
    y_map = {d: i for i, d in enumerate(all_diseases)}

    for _, row in bubble_df.iterrows():
        x = row["atom"]
        y = y_map[row["disease"]]
        s = row["strength"] * 300
        c = cat_col.get(row["category"], "#aaa")
        ax_D.scatter(x, y, s=s, c=c, alpha=0.75, edgecolors="white", linewidths=0.5)

    ax_D.set_xlim(0.3, K + 0.7)
    ax_D.set_ylim(-0.7, len(all_diseases) - 0.3)
    ax_D.set_xticks(range(1, K + 1))
    ax_D.set_xticklabels([f"A{k+1}" for k in range(K)], fontsize=8)
    ax_D.set_yticks(range(len(all_diseases)))
    ax_D.set_yticklabels(all_diseases, fontsize=7)
    ax_D.set_xlabel("Atom", fontsize=8)
    ax_D.grid(True, alpha=0.2, linestyle="--")
    ax_D.set_facecolor("#fafafa")

    # Legend
    legend_elements = [
        mpatches.Patch(color=cat_col["cancer"],       label="Cancer"),
        mpatches.Patch(color=cat_col["fibrosis"],     label="Fibrosis"),
        mpatches.Patch(color=cat_col["inflammation"], label="Inflammation"),
        mpatches.Patch(color=cat_col["homeostasis"],  label="Homeostasis"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#aaa",
               markersize=8,  label="Documented (●)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#aaa",
               markersize=5,  label="Inferred (●)"),
    ]
    ax_D.legend(handles=legend_elements, fontsize=6.5, loc="lower right",
                framealpha=0.9, ncol=2)

    # ── Panel E: Trajectory UMAP (full width, bottom row) ────────────────────
    ax_E = fig.add_subplot(outer[2, :])
    fig.text(0.06, _y_label(2), "E", fontsize=18, fontweight="bold", va="top")
    fig.text(0.08, _y_label(2),
             "Disease Trajectory Map (UMAP of Atom-Weight Space)",
             fontsize=9, va="top", color="#333")

    SRC_COLORS_E = {
        "pbmc3k":         "#2196F3",
        "pbmc68k":        "#64B5F6",
        "synthetic_covid":"#FF9800",
        "synthetic_ipf":  "#795548",
        "synthetic_tumor":"#F44336",
    }
    SRC_LABELS_E = {
        "pbmc3k":         "Healthy PBMC (3k)",
        "pbmc68k":        "Healthy PBMC (68k)",
        "synthetic_covid":"COVID-19",
        "synthetic_ipf":  "IPF/Fibrosis",
        "synthetic_tumor":"Tumor TME",
    }
    DISEASE_GROUPS_E = {
        "healthy":      ["pbmc3k","pbmc68k"],
        "inflammation": ["synthetic_covid"],
        "fibrosis":     ["synthetic_ipf"],
        "cancer":       ["synthetic_tumor"],
    }

    if traj_ok:
        progression = meta_pt["progression_score"].values

        # Scatter
        for src in meta_pt["dataset_source"].unique():
            mask = (meta_pt["dataset_source"] == src).values
            ax_E.scatter(embedding[mask, 0], embedding[mask, 1],
                         c=SRC_COLORS_E.get(src, "#aaa"),
                         label=SRC_LABELS_E.get(src, src),
                         s=55, edgecolors="white", linewidths=0.5, zorder=3)

        # Group centroids
        centroids_E = {}
        for group, srcs in DISEASE_GROUPS_E.items():
            mask = meta_pt["dataset_source"].isin(srcs).values
            if mask.any():
                centroids_E[group] = embedding[mask].mean(axis=0)

        # Arrows healthy → disease
        arrow_defs = {
            "cancer":       ("#F44336", "Tumor TME"),
            "fibrosis":     ("#795548", "IPF/Fibrosis"),
            "inflammation": ("#FF9800", "COVID-19"),
        }
        hc = centroids_E["healthy"]
        ax_E.scatter(*hc, marker="*", s=220, c="black", zorder=5)
        ax_E.text(hc[0]+0.15, hc[1]-0.25, "Healthy\ncentroid",
                  fontsize=7.5, color="black", fontweight="bold")
        for group, (color, lbl) in arrow_defs.items():
            if group in centroids_E:
                dc = centroids_E[group]
                ax_E.annotate("", xy=dc, xytext=hc,
                              arrowprops=dict(arrowstyle="-|>", color=color,
                                              lw=2.5, mutation_scale=16))
                mid = (hc + dc) / 2 + np.array([0.1, 0.1])
                ax_E.text(mid[0], mid[1], lbl, fontsize=8, color=color,
                          fontweight="bold", ha="center")

        # Colorbar for progression score
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=progression.min(), vmax=progression.max())
        sm   = ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_E, label="Disease progression score",
                     shrink=0.8, pad=0.01)

        ax_E.set_xlabel("UMAP-1", fontsize=8)
        ax_E.set_ylabel("UMAP-2", fontsize=8)
        ax_E.legend(fontsize=7.5, loc="lower right", framealpha=0.9, ncol=5)
        ax_E.grid(True, alpha=0.2)
        ax_E.set_title("", pad=0)
    else:
        ax_E.text(0.5, 0.5, "Trajectory data not found\n(run pseudotime.py first)",
                  ha="center", va="center", fontsize=12, transform=ax_E.transAxes)
        ax_E.set_axis_off()

    # ── Panel F: EHR Bridge — predicted vs actual atom weights ───────────────
    ax_F0 = fig.add_subplot(outer[3, 0])
    ax_F1 = fig.add_subplot(outer[3, 1])
    fig.text(0.06, _y_label(3), "F", fontsize=18, fontweight="bold", va="top")
    fig.text(0.08, _y_label(3),
             "EHR Bridge: Blood Panel → Atom Weight Prediction (no scRNA-seq)",
             fontsize=9, va="top", color="#333")

    if ehr_ok:
        from sklearn.metrics import r2_score as _r2
        for ax_sub, atom_idx, atom_name, color in [
            (ax_F0, 0, "Atom 1 (Cancer/Tumor TME)",    "#d62728"),
            (ax_F1, 1, "Atom 2 (Fibrosis/IPF)",        "#8c564b"),
        ]:
            y_true = _Yte[:, atom_idx]
            y_pred = _Ypred[:, atom_idx]
            r2     = _r2(y_true, y_pred)
            mn     = min(y_true.min(), y_pred.min()) - 0.02
            mx     = max(y_true.max(), y_pred.max()) + 0.02
            ax_sub.scatter(y_true, y_pred, c=color, s=60,
                           edgecolors="white", linewidths=0.5, alpha=0.85)
            ax_sub.plot([mn, mx], [mn, mx], "k--", lw=1.2, alpha=0.5)
            ax_sub.set_xlabel("Actual atom weight (scRNA-seq)", fontsize=8)
            ax_sub.set_ylabel("Predicted atom weight (blood panel)", fontsize=8)
            ax_sub.set_title(f"{atom_name}\nR² = {r2:.3f}", fontsize=9)
            ax_sub.text(0.05, 0.93, f"R² = {r2:.3f}", transform=ax_sub.transAxes,
                        fontsize=10, fontweight="bold", color=color, va="top")
            ax_sub.grid(True, alpha=0.2)

        # R² bar for all atoms
        r2_vals = ehr_r2["r2"].values
        cats    = ehr_r2["category"].values
        cat_colors_bar = {"cancer":"#d62728","fibrosis":"#8c564b",
                          "inflammation":"#ff7f0e","homeostasis":"#2196F3","unknown":"#aaa"}
        bar_colors = [cat_colors_bar.get(c,"#aaa") for c in cats]
    else:
        for ax_sub in [ax_F0, ax_F1]:
            ax_sub.text(0.5, 0.5, "EHR data not found", ha="center", va="center",
                        transform=ax_sub.transAxes, fontsize=11)
            ax_sub.set_axis_off()

    # ── Panel G: Drug repurposing — honest ChEMBL results with p-values ────────
    ax_G = fig.add_subplot(outer[4, :])
    fig.text(0.06, _y_label(4), "G", fontsize=18, fontweight="bold", va="top")
    fig.text(0.08, _y_label(4),
             "Drug Repurposing — ChEMBL External API Query  "
             "[PROOF-OF-CONCEPT: p>0.05 at n=40; 9/9 curated result RETRACTED]",
             fontsize=9, va="top", color="#c0392b")

    # Load honest ChEMBL results
    try:
        drug_summary_path = f"{DATA_DIR}/drug_repurposing_summary.csv"
        drug_sum_df  = pd.read_csv(drug_summary_path)
        drug_ok2     = True
    except FileNotFoundError:
        drug_ok2 = False

    if drug_ok2 and len(drug_sum_df) > 0:
        # Bar chart: GT hits / total GT, with p-value annotations
        inner_G = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4, :],
                                                    wspace=0.45)
        ax_G.set_visible(False)
        ax_G1 = fig.add_subplot(inner_G[0])
        ax_G2 = fig.add_subplot(inner_G[1])

        cat_colors_g = {"cancer":"#d62728","fibrosis":"#8c564b","inflammation":"#ff7f0e"}
        atoms_g = drug_sum_df["atom"].tolist()
        cats_g  = drug_sum_df["category"].tolist()
        hit_rates = (drug_sum_df["gt_hits"] / drug_sum_df["gt_total"]).tolist()
        pvals_g   = drug_sum_df["pval"].tolist()
        n_mols_g  = drug_sum_df["mol_ids"].tolist()

        x = np.arange(len(atoms_g))
        bars = ax_G1.bar(x, hit_rates,
                         color=[cat_colors_g.get(c,"#aaa") for c in cats_g],
                         edgecolor="white", linewidth=0.5, alpha=0.85)
        ax_G1.axhline(0.5, color="#333", lw=1, ls="--", alpha=0.5, label="50% hit rate")
        for i, (p, h, nm) in enumerate(zip(pvals_g, hit_rates, n_mols_g)):
            sig_str = f"p={p:.2f}" + (" *" if p < 0.05 else " ns")
            ax_G1.text(i, h + 0.03, sig_str, ha="center", fontsize=8,
                       color="red" if p < 0.05 else "#555")
            ax_G1.text(i, -0.08, f"n={nm}", ha="center", fontsize=7, color="#777")
        ax_G1.set_xticks(x)
        ax_G1.set_xticklabels([f"A{a}\n({c[:5]})" for a,c in zip(atoms_g, cats_g)], fontsize=8)
        ax_G1.set_ylabel("GT drug hit rate", fontsize=8)
        ax_G1.set_ylim(-0.15, 1.0)
        ax_G1.set_title("Ground-truth drug recovery\n(ChEMBL external; p>0.05 = not significant)", fontsize=8)
        ax_G1.grid(True, alpha=0.2, axis="y")

        # Table of GT found vs not found
        ax_G2.axis("off")
        table_data = []
        for _, r in drug_sum_df.iterrows():
            found   = r["gt_found"] if r["gt_found"] else "none"
            not_fd  = len(r.get("gt_found","").split(",")) if r["gt_found"] else 0
            status  = r.get("status","?")
            table_data.append([
                f"A{r['atom']} {r['category'][:5]}",
                f"{r['gt_hits']}/{r['gt_total']}",
                found[:25] if found != "none" else "—",
                f"p={r['pval']:.2f}",
                status,
            ])
        tbl = ax_G2.table(
            cellText=table_data,
            colLabels=["Atom", "GT hits", "Drugs found", "p-val", "Status"],
            loc="center", cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.6)
        ax_G2.set_title("ChEMBL query summary\n(RETRACTED: 9/9 curated result was circular)",
                         fontsize=8, color="#c0392b")
    else:
        ax_G.text(0.5, 0.5,
                  "Drug repurposing data not found\n"
                  "(run drug_repurposing.py — uses ChEMBL external API)",
                  ha="center", va="center", transform=ax_G.transAxes, fontsize=11)
        ax_G.set_axis_off()

    # ── Panel H: Spatial healer vs non-healer ─────────────────────────────────
    ax_H = fig.add_subplot(outer[5, :])
    fig.text(0.06, _y_label(5), "H", fontsize=18, fontweight="bold", va="top")
    fig.text(0.08, _y_label(5),
             "Spatial Communication Atlas — DFU Wound Cross-Section (20×20 grid)",
             fontsize=9, va="top", color="#333")

    if spatial_ok:
        from matplotlib.image import imread as _imread
        spatial_img = _imread(spatial_img_path)
        ax_H.imshow(spatial_img, aspect="auto")
        ax_H.set_axis_off()
    else:
        ax_H.text(0.5, 0.5, "Spatial map not found\n(run spatial_mapping.py)",
                  ha="center", va="center", transform=ax_H.transAxes, fontsize=11)
        ax_H.set_axis_off()

    # ── Save ──────────────────────────────────────────────────────────────────
    out = f"{FIG_DIR}/periodic_table_figure.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    make_figure()
