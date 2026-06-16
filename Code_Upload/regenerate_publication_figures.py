from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PUB_DIR = ROOT / "Publication_Figures"
PAIN_DIR = ROOT / "Pain_V7_Results"
DEHY_DIR = ROOT / "Dehydration_V7_Results"

TOTAL_WINDOWS = 1414
PAIN_POSITIVE_WINDOWS = 364
DEHY_POSITIVE_WINDOWS = 800
PAIN_LOPO_FOLDS = 8
DEHY_LOPO_FOLDS = 12

PUB_DIR.mkdir(parents=True, exist_ok=True)


def save_dual(fig: plt.Figure, name: str) -> None:
    fig.savefig(PUB_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PUB_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PUB_DIR / f"{name}.tif", dpi=300, bbox_inches="tight", format="tiff")
    plt.close(fig)


def sync_figure(src: Path, dst_name: str) -> None:
    tif_dst = PUB_DIR / f"{dst_name}.tif"
    png_dst = PUB_DIR / f"{dst_name}.png"

    shutil.copy2(src, tif_dst)

    src_png = src.with_suffix(".png")
    if src_png.exists():
        shutil.copy2(src_png, png_dst)
    else:
        image = plt.imread(src)
        plt.imsave(png_dst, image)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — Study Design Flowchart (CONSORT-style, V6 layout)
# ═══════════════════════════════════════════════════════════════════════
def draw_consort() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    box_kw  = dict(boxstyle="round,pad=0.4", facecolor="#E8F0FE",
                   edgecolor="#333333", linewidth=1.2)
    box_kw2 = dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0",
                   edgecolor="#333333", linewidth=1.2)
    box_kw3 = dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                   edgecolor="#333333", linewidth=1.2)
    excl_kw = dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE",
                   edgecolor="#333", linewidth=1.2)
    txt_kw  = dict(ha="center", va="center", fontsize=9, fontweight="normal")
    arrow   = dict(arrowstyle="->", color="#333", lw=1.5)

    # Row 1 — Screening
    ax.text(5, 15,
            "Nursing home residents screened\n(n\u2248100)",
            bbox=box_kw, **txt_kw)

    ax.annotate("", xy=(5, 13.8), xytext=(5, 14.5), arrowprops=arrow)

    # Screening exclusion
    ax.text(8.2, 14.2,
            "Did not meet\ninclusion criteria (n\u224860)",
            bbox=excl_kw, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(6.6, 14.2), xytext=(5.4, 14.5), arrowprops=arrow)

    # Row 2 — Eligible
    ax.text(5, 13.3,
            "Assessed for eligibility (n\u224840)",
            bbox=box_kw, **txt_kw)

    ax.annotate("", xy=(5, 12.3), xytext=(5, 12.8), arrowprops=arrow)

    # Declined exclusion
    ax.text(8.2, 12.6,
            "Declined early (n\u224816)",
            bbox=excl_kw, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(6.6, 12.6), xytext=(5.4, 12.8), arrowprops=arrow)

    # Row 3 — Consented
    ax.text(5, 11.8,
            "Consented and enrolled (n=24)",
            bbox=box_kw, **txt_kw)

    ax.annotate("", xy=(5, 10.8), xytext=(5, 11.3), arrowprops=arrow)

    # Row 4 — Enrolled & equipped
    ax.text(5, 10.3,
            "Completed monitoring with\nApple Watch Ultra 2 (n=16)",
            bbox=box_kw, **txt_kw)

    # Bad data quality exclusion
    ax.text(8.2, 11.05,
            "Did not complete\nmonitoring (n=8)",
            bbox=excl_kw, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(6.6, 11.05), xytext=(5.4, 11.3), arrowprops=arrow)

    ax.annotate("", xy=(5, 9.2), xytext=(5, 9.8), arrowprops=arrow)

    # Row 5 — Monitoring
    ax.text(5, 8.7,
            "4-week continuous monitoring\n"
            "3\u00d7 weekly clinical assessments\n"
            "(VAS pain, dehydration signs, vitals)",
            bbox=box_kw2, **txt_kw)

    ax.annotate("", xy=(5, 7.4), xytext=(5, 8.1), arrowprops=arrow)

    # Row 6 — Data Processing
    ax.text(5, 6.9,
            "Data processing\n"
            "\u2022 120-min time windows  \u2022 Feature aggregation\n"
            "\u2022 Gait imputation  \u2022 Lag & rolling features",
            bbox=box_kw2, ha="center", va="center", fontsize=8.5)

    # Quality filter exclusion box
    ax.text(8.5, 6.0,
            "Excluded (low data quality):\n"
            "3 patients (IDs 9, 10, 12)\n"
            "(< 30% of median windows)",
            bbox=excl_kw, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(6.9, 6.0), xytext=(5.5, 6.4), arrowprops=arrow)

    # Arrows split
    ax.annotate("", xy=(3, 5.0), xytext=(5, 6.3), arrowprops=arrow)
    ax.annotate("", xy=(7, 5.0), xytext=(5, 6.3), arrowprops=arrow)

    # Row 7 — Pain branch
    n_pain, n_pain_pos, n_pain_pat = TOTAL_WINDOWS, PAIN_POSITIVE_WINDOWS, 13
    ax.text(3, 4.5,
            f"Pain Detection\n"
            f"n={n_pain} time windows ({n_pain_pat} patients)\n"
            f"{n_pain_pos} positive ({100*n_pain_pos/n_pain:.0f}%)",
            bbox=box_kw3, **txt_kw)

    # Row 7 — Dehydration branch
    n_dehy, n_dehy_pos, n_dehy_pat = TOTAL_WINDOWS, DEHY_POSITIVE_WINDOWS, 13
    ax.text(7, 4.5,
            f"Dehydration Detection\n"
            f"n={n_dehy} time windows ({n_dehy_pat} patients)\n"
            f"{n_dehy_pos} positive ({100*n_dehy_pos/n_dehy:.0f}%)",
            bbox=box_kw3, **txt_kw)

    # Arrows down
    ax.annotate("", xy=(3, 3.0), xytext=(3, 3.8), arrowprops=arrow)
    ax.annotate("", xy=(7, 3.0), xytext=(7, 3.8), arrowprops=arrow)

    # Row 8 — LOPOCV
    ax.text(3, 2.5,
            "Leave-One-Patient-Out CV\n"
            f"{PAIN_LOPO_FOLDS} evaluable folds\n"
            "XGBoost",
            bbox=box_kw2, ha="center", va="center", fontsize=8.5)

    ax.text(7, 2.5,
            "Leave-One-Patient-Out CV\n"
            f"{DEHY_LOPO_FOLDS} evaluable folds\n"
            "XGBoost",
            bbox=box_kw2, ha="center", va="center", fontsize=8.5)

    # Arrows to evaluation
    ax.annotate("", xy=(3, 1.2), xytext=(3, 1.8), arrowprops=arrow)
    ax.annotate("", xy=(7, 1.2), xytext=(7, 1.8), arrowprops=arrow)

    # Row 9 — Evaluation
    ax.text(3, 0.7,
            "Evaluation\n"
            "Pooled LOPO AUC, Accuracy, F1\n"
            "SHAP Interpretability",
            bbox=box_kw3, ha="center", va="center", fontsize=8.5)

    ax.text(7, 0.7,
            "Evaluation\n"
            "Pooled LOPO AUC, Accuracy, F1\n"
            "SHAP Interpretability",
            bbox=box_kw3, ha="center", va="center", fontsize=8.5)

    fig.suptitle("Figure 1: Study Design and Analysis Pipeline",
                 fontsize=13, fontweight="bold", y=0.98)
    save_dual(fig, "Fig1_CONSORT")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — Label Distribution
# ═══════════════════════════════════════════════════════════════════════
def draw_label_distribution() -> None:
    script_path = ROOT / "regenerate_all_figures.py"
    source = script_path.read_text(encoding="utf-8")
    prep_source = source.split(
        "# ═══════════════════════════════════════════════════════════════════════\n"
        "# 5. LOPOCV + FIGURE GENERATION"
    )[0]
    namespace = {"__file__": str(script_path), "__name__": "__prep_only__"}
    exec(prep_source, namespace)
    df_pain_final = namespace["prepare_pain_data"]()
    df_dehy_final = namespace["prepare_dehydration_data"]()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=False)
    colors = {
        "negative": "#0072B2",  # Okabe-Ito blue
        "positive": "#D55E00",  # Okabe-Ito vermillion
    }
    width = 0.38

    ax = axes[0]
    pain_counts = df_pain_final.groupby("ID")["pain_label"].value_counts().unstack(fill_value=0)
    for label in (0.0, 1.0):
        if label not in pain_counts.columns:
            pain_counts[label] = 0
    pain_counts = pain_counts[[0.0, 1.0]].sort_index()
    x = np.arange(len(pain_counts))
    ax.bar(x - width / 2, pain_counts[0.0], width, label="No pain (0)",
           color=colors["negative"], edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, pain_counts[1.0], width, label="Pain (1)",
           color=colors["positive"], edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Patient ID", fontsize=14)
    ax.set_ylabel("Number of time windows", fontsize=14)
    ax.set_title("A. Pain label distribution", fontsize=16, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ID{i}" for i in pain_counts.index], rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, patient_id in enumerate(pain_counts.index):
        if pain_counts.loc[patient_id, 0.0] == 0 or pain_counts.loc[patient_id, 1.0] == 0:
            ax.text(i, pain_counts.loc[patient_id].max() + 2, "*", ha="center",
                    fontsize=18, color="#CC79A7", fontweight="bold")

    ax = axes[1]
    dehy_counts = df_dehy_final.groupby("ID")["dehydration_label"].value_counts().unstack(fill_value=0)
    for label in (0.0, 1.0):
        if label not in dehy_counts.columns:
            dehy_counts[label] = 0
    dehy_counts = dehy_counts[[0.0, 1.0]].sort_index()
    x = np.arange(len(dehy_counts))
    ax.bar(x - width / 2, dehy_counts[0.0], width, label="Not dehydrated (0)",
           color=colors["negative"], edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, dehy_counts[1.0], width, label="Dehydrated (1)",
           color=colors["positive"], edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Patient ID", fontsize=14)
    ax.set_ylabel("Number of time windows", fontsize=14)
    ax.set_title("B. Dehydration label distribution", fontsize=16, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ID{i}" for i in dehy_counts.index], rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, patient_id in enumerate(dehy_counts.index):
        if dehy_counts.loc[patient_id, 0.0] == 0 or dehy_counts.loc[patient_id, 1.0] == 0:
            ax.text(i, dehy_counts.loc[patient_id].max() + 2, "*", ha="center",
                    fontsize=18, color="#CC79A7", fontweight="bold")

    fig.suptitle("Figure 2: Per-patient label distribution (* = single-class patient)",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_dual(fig, "Fig2_Label_Distribution")


def draw_auc_boxplots() -> None:
    pain = pd.read_csv(PAIN_DIR / "lopo_per_patient.csv")
    dehy = pd.read_csv(DEHY_DIR / "lopo_per_patient.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    data = [pain["AUC"].dropna().values, dehy["AUC"].dropna().values]

    bp = ax.boxplot(data, tick_labels=["Pain", "Dehydration"], patch_artist=True, widths=0.55)
    colors = ["#66c2a5", "#fc8d62"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for i, arr in enumerate(data, start=1):
        jitter = np.random.default_rng(42 + i).normal(0, 0.04, size=len(arr))
        ax.scatter(np.full_like(arr, i, dtype=float) + jitter, arr, alpha=0.75, s=26, color="#444444")
        ax.text(i, np.mean(arr) + 0.02, f"mean={np.mean(arr):.3f}", ha="center", fontsize=9)

    ax.axhline(0.5, linestyle="--", color="#666666", linewidth=1.0)
    ax.set_ylabel("Per-patient AUC (LOPO)")
    ax.set_ylim(0.3, 0.8)
    ax.set_title("Per-patient AUC Distribution Across LOPO Folds")
    ax.grid(axis="y", alpha=0.25)
    save_dual(fig, "Fig2b_Score_Boxplots")


def copy_latest_result_figures() -> None:
    mapping = {
        PAIN_DIR / "ROC_Curve_pain_CV_averaged.tif": "Fig3_ROC_Pain",
        DEHY_DIR / "ROC_Curve_dehydration_CV_averaged.tif": "Fig4_ROC_Dehydration",
        PAIN_DIR / "Shap_values_pain.tif": "Fig5_SHAP_Pain",
        DEHY_DIR / "Shap_values_dehydration.tif": "Fig6_SHAP_Dehydration",
        PAIN_DIR / "ROC_Curve_pain.tif": "Supplement_FigS1_ROC_Pain_BestSplit",
        DEHY_DIR / "ROC_Curve_dehydration.tif": "Supplement_FigS2_ROC_Dehydration_BestSplit",
    }

    for src, dst_name in mapping.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source figure: {src}")
        sync_figure(src, dst_name)


def update_metrics_csv() -> None:
    pain_metrics = json.loads((PAIN_DIR / "test_metrics.json").read_text(encoding="utf-8"))
    dehy_metrics = json.loads((DEHY_DIR / "test_metrics.json").read_text(encoding="utf-8"))

    rows = [
        ("Accuracy", pain_metrics["pooled_Accuracy"], dehy_metrics["pooled_Accuracy"]),
        ("Precision", pain_metrics["pooled_Precision"], dehy_metrics["pooled_Precision"]),
        ("Recall", pain_metrics["pooled_Recall"], dehy_metrics["pooled_Recall"]),
        ("F1 Score", pain_metrics["pooled_F1"], dehy_metrics["pooled_F1"]),
        ("AUC (Pooled)", pain_metrics["pooled_AUC"], dehy_metrics["pooled_AUC"]),
        ("AUC (Per-patient Mean)", pain_metrics["per_patient_mean_AUC"], dehy_metrics["per_patient_mean_AUC"]),
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Pain", "Dehydration"])
    df[["Pain", "Dehydration"]] = df[["Pain", "Dehydration"]].round(3)
    df.to_csv(PUB_DIR / "metrics_comparison.csv", index=False)


def main() -> None:
    draw_consort()
    draw_label_distribution()
    draw_auc_boxplots()
    copy_latest_result_figures()
    update_metrics_csv()
    print(f"Updated publication figures in: {PUB_DIR}")


if __name__ == "__main__":
    main()
