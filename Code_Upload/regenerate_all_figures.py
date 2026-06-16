"""
Regenerate ALL publication figures with unified styling.

Produces:
  Fig3_ROC_Pain          – Averaged held-out ROC (pain, 8 LOPO folds)
  Fig4_ROC_Dehydration   – Averaged held-out ROC (dehydration, 12 LOPO folds)
  Fig_ROC_Pain_Female    – Averaged held-out ROC (female-only pain, 7 LOPO folds)
  Supplement_FigS1_ROC_Pain_BestSplit       – Best-fold ROC (pain)
  Supplement_FigS2_ROC_Dehydration_BestSplit – Best-fold ROC (dehydration)
  Fig5_SHAP_Pain         – SHAP beeswarm (pain)
  Fig6_SHAP_Dehydration  – SHAP beeswarm (dehydration)

All ROC figures use identical axes, fonts, colours, and layout.
Both PNG and TIF are rendered from the same matplotlib figure.
"""
from __future__ import annotations

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc as sklearn_auc,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
PUB_DIR = "Publication_Figures"
os.makedirs(PUB_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Unified plotting helpers
# ═══════════════════════════════════════════════════════════════════════
FIGSIZE = (6, 6)
DPI = 300
FONT_TITLE = 13
FONT_LABEL = 11
FONT_LEGEND = 9.5
FONT_TICK = 10


FEATURE_NAME_MAP = {
    "ActiveEnergyBurned": "Active energy",
    "BasalEnergyBurned": "Basal energy",
    "DistanceWalkingRunning": "Walking/running distance",
    "HeartRate": "HR",
    "HeartRateVariabilitySDNN": "HRV-SDNN",
    "OxygenSaturation": "SpO2",
    "PhysicalEffort": "Physical effort",
    "RestingHeartRate": "Resting HR",
    "SixMinuteWalkTestDistance": "Six-minute walk distance",
    "StairAscentSpeed": "Stair ascent speed",
    "StairDescentSpeed": "Stair descent speed",
    "StepCount": "Step count",
    "WalkingAsymmetry": "Walking asymmetry",
    "WalkingDoubleSupportPercentage": "Walking double support",
    "WalkingHeartRateAverage": "Walking HR average",
    "WalkingSpeed": "Walking speed",
}

STAT_NAME_MAP = {
    "max": "maximum",
    "median": "median",
    "min": "minimum",
    "sum": "sum",
}

WINDOW_HOURS = 2


def format_feature_name(feature_name: str) -> str:
    """Convert raw model feature columns into publication-friendly labels."""
    name = feature_name
    prefix = ""
    lag_label = ""

    if name.startswith("diff_"):
        prefix = "Change in "
        name = name[5:]

    for suffix in ("_lag1", "_lag3", "_lag6"):
        if name.endswith(suffix):
            hours_ago = int(suffix[-1]) * WINDOW_HOURS
            lag_label = f" ({hours_ago} h ago)"
            name = name[: -len(suffix)]
            break

    if name.endswith("_sum_sum_lag"):
        name = name[: -len("_sum_sum_lag")]
        lag_label = " (rolling sum)"

    parts = name.split("_")
    base = FEATURE_NAME_MAP.get(parts[0], parts[0])
    stats = [STAT_NAME_MAP.get(part, part) for part in parts[1:] if part]
    stat_label = f" ({', '.join(stats)})" if stats else ""
    return f"{prefix}{base}{stat_label}{lag_label}"


def _setup_roc_axes(ax, title: str) -> None:
    """Apply identical formatting to every ROC axis."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=FONT_LABEL)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, pad=10)
    ax.legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FONT_TICK)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="_nolegend_")


def _save_fig(fig, name: str) -> None:
    """Save a figure as both PNG and TIF from the same render."""
    path_png = os.path.join(PUB_DIR, f"{name}.png")
    path_tif = os.path.join(PUB_DIR, f"{name}.tif")
    fig.savefig(path_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(path_tif, dpi=DPI, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"  Saved {name}.png + .tif")


def plot_averaged_roc(
    mean_fpr: np.ndarray,
    tpr_curves: list[np.ndarray],
    title: str,
    fig_name: str,
    line_color: str,
    n_folds_label: str,
) -> tuple[float, float]:
    """Averaged held-out ROC with individual fold traces and ±1 SD band."""
    tprs = np.asarray(tpr_curves)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    auc_values = np.array([sklearn_auc(mean_fpr, t) for t in tprs])
    mean_auc = sklearn_auc(mean_fpr, mean_tpr)
    std_auc = auc_values.std()
    std_tpr = tprs.std(axis=0)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Individual fold curves
    for t in tprs:
        ax.plot(mean_fpr, t, alpha=0.12, linewidth=0.8, color=line_color)

    # Mean curve
    ax.plot(
        mean_fpr, mean_tpr, color=line_color, linewidth=2.2,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
    )

    # ±1 SD band
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.18, color=line_color, label="± 1 SD",
    )

    # Chance line
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Chance")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=FONT_LABEL)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, pad=10)
    ax.legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout()
    _save_fig(fig, fig_name)

    print(f"    AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    return mean_auc, std_auc


def plot_best_fold_roc(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str,
    fig_name: str,
    n_bootstraps: int = 200,
) -> tuple[float, float]:
    """Supplementary best-fold ROC: internal 3-fold CV + held-out test + bootstrap CI."""

    # Internal 3-fold CV on training data
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_proba = np.zeros(len(y_train))
    for train_idx, val_idx in kf.split(X_train, y_train):
        m = clone(model)
        try:
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        except Exception:
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        cv_proba[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
    fpr_cv, tpr_cv, _ = roc_curve(y_train, cv_proba)
    auc_cv = sklearn_auc(fpr_cv, tpr_cv)

    # Held-out test
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba)
    auc_test = sklearn_auc(fpr_test, tpr_test)

    # Bootstrap CI on test
    rng = np.random.RandomState(42)
    mean_fpr = np.linspace(0, 1, 200)
    boot_tprs = []
    y_test_np = np.asarray(y_test)
    y_proba_np = np.asarray(y_pred_proba)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_test_np), len(y_test_np))
        if np.unique(y_test_np[idx]).size < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_test_np[idx], y_proba_np[idx])
        boot_tprs.append(np.interp(mean_fpr, fpr_b, tpr_b))
    boot_tprs = np.array(boot_tprs)
    tpr_lo = np.percentile(boot_tprs, 5, axis=0)
    tpr_hi = np.percentile(boot_tprs, 95, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Chance")
    ax.plot(fpr_cv, tpr_cv, color="#1976D2", lw=2,
            label=f"3-fold CV (AUC = {auc_cv:.2f})")
    ax.plot(fpr_test, tpr_test, color="#D32F2F", lw=2,
            label=f"Held-out test (AUC = {auc_test:.2f})")
    ax.fill_between(mean_fpr, tpr_lo, tpr_hi, color="#D32F2F", alpha=0.12,
                    label="Test 90 % CI (bootstrap)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=FONT_LABEL)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, pad=10)
    ax.legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout()
    _save_fig(fig, fig_name)

    print(f"    CV AUC = {auc_cv:.4f}, Test AUC = {auc_test:.4f}")
    return auc_cv, auc_test


def plot_shap_beeswarm(
    model,
    X_test: pd.DataFrame,
    title: str,
    fig_name: str,
    max_display: int = 20,
) -> None:
    """Consistently-formatted SHAP beeswarm plot."""
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)
    X_display = X_test.copy()
    X_display.columns = [format_feature_name(col) for col in X_display.columns]

    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    shap.summary_plot(
        shap_values.values, X_display, show=False, max_display=max_display,
        plot_size=None,
    )
    ax = plt.gca()
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    # Ensure tick labels don't overlap
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    plt.tight_layout()

    path_png = os.path.join(PUB_DIR, f"{fig_name}.png")
    path_tif = os.path.join(PUB_DIR, f"{fig_name}.tif")
    plt.savefig(path_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(path_tif, dpi=DPI, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"  Saved {fig_name}.png + .tif")


# ═══════════════════════════════════════════════════════════════════════
# Pipeline helpers
# ═══════════════════════════════════════════════════════════════════════
VALID_XGB_PARAMS = {
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "gamma",
    "reg_alpha", "reg_lambda", "max_leaves", "grow_policy",
    "scale_pos_weight", "base_score", "booster", "n_jobs",
}


def build_pipeline(y_train=None, k_neighbors=3, **xgb_params):
    """XGBoost + SMOTE pipeline (auto-adjusts k_neighbors for small sets)."""
    filtered = {k: v for k, v in xgb_params.items() if k in VALID_XGB_PARAMS}

    if y_train is not None:
        class_counts = y_train.value_counts()
        minority_count = int(class_counts.min()) if not class_counts.empty else 0
        if minority_count < 2:
            smote = "passthrough"
        else:
            smote = SMOTE(random_state=42,
                          k_neighbors=min(k_neighbors, minority_count - 1))
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", smote),
        ("classifier", XGBClassifier(
            random_state=42, eval_metric="logloss", **filtered)),
    ])


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING (mirrors TexHype_V7 cells 8-30)
# ═══════════════════════════════════════════════════════════════════════
print("Loading data …")
all_data = pd.read_csv("output_all.csv")
personal_data = pd.read_csv("personal_data.csv")

year_of_measurement = 2024
personal_data["Age"] = year_of_measurement - personal_data["Year of Birth"]
personal_data["Gender"] = personal_data["Sex"].map(
    {"HKBiologicalSexFemale": 0, "HKBiologicalSexMale": 1}
)

all_data = all_data.pivot_table(
    index=["ID", "Start", "End", "enrollment_date", "final_visit"],
    columns="Name", values="Value", aggfunc="mean",
).reset_index()
df = all_data.merge(personal_data[["ID", "Gender", "Age"]], on="ID", how="left")

df.columns = df.columns.str.replace("HKCategoryTypeIdentifier", "", regex=True)
df.columns = df.columns.str.replace("HKQuantityTypeIdentifier", "", regex=True)
df[["Height", "BodyMass"]] = df.groupby("ID")[["Height", "BodyMass"]].ffill().bfill()

date_cols = ["Start", "End", "enrollment_date", "final_visit"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
mask = (df["Start"].dt.date >= df["enrollment_date"].dt.date) & (
    df["End"].dt.date <= df["final_visit"].dt.date)
df = df.loc[mask].copy()
df = df.drop(columns=["enrollment_date", "final_visit"]).reset_index(drop=True)

duration = 120
df["Start"] = pd.to_datetime(df["Start"], errors="coerce")
df = df.sort_values(["ID", "Start"]).copy()
tw = pd.Timedelta(minutes=duration)
hr = df[df["HeartRate"].notna()]
first_ = hr.groupby("ID")["Start"].min()
last_ = hr.groupby("ID")["Start"].max()

tw_list = []
for id_ in df["ID"].unique():
    if id_ not in first_.index:
        continue
    start_time = first_[id_]
    end_time = last_[id_]
    n_windows = int(((end_time - start_time).total_seconds()) // (duration * 60)) + 1
    for tw_num in range(n_windows):
        window_start = start_time + pd.Timedelta(minutes=tw_num * duration)
        window_end = window_start + tw
        tw_list.append((id_, tw_num, window_start, window_end))
tw_df = pd.DataFrame(tw_list, columns=["ID", "time_window", "window_start", "window_end"])

df["_start"] = df["ID"].map(first_)
df["_end"] = df["ID"].map(last_)
mask = (df["Start"] >= df["_start"]) & (df["Start"] <= df["_end"])
df = df.loc[mask].copy()
delta = (df["Start"] - df["_start"]).dt.total_seconds()
df["time_window"] = (delta // (duration * 60)).astype("Int64")
df.drop(columns=["_start", "_end"], inplace=True)

group = df.groupby(["ID", "time_window"])
df = group.agg({
    "Age": "first", "Gender": "first",
    "Height": "first", "BodyMass": "first",
    "HeartRate": ["max", "median", "min", "std"],
    "OxygenSaturation": ["max", "median", "min"],
    "HeartRateVariabilitySDNN": ["max", "median", "min"],
    "RestingHeartRate": ["median"],
    "ActiveEnergyBurned": ["max", "median", "min", "sum"],
    "BasalEnergyBurned": ["max", "median", "min", "sum"],
    "PhysicalEffort": ["max", "median", "min", "sum"],
    "AppleStandHour": ["sum"], "AppleStandTime": ["sum"],
    "AppleExerciseTime": ["sum"], "DistanceWalkingRunning": ["sum"],
    "StepCount": ["sum"], "WalkingStepLength": ["median"],
    "WalkingSpeed": ["median"], "WalkingAsymmetryPercentage": ["median"],
    "WalkingDoubleSupportPercentage": ["median"],
    "WalkingHeartRateAverage": ["median"], "FlightsClimbed": ["sum"],
    "StairAscentSpeed": ["median"], "StairDescentSpeed": ["median"],
    "SixMinuteWalkTestDistance": ["median"],
}).reset_index()

df.columns = [
    col[0] if isinstance(col, tuple) and col[1] == ""
    else f"{col[0]}_{col[1]}" for col in df.columns
]
df = pd.merge(tw_df, df, on=["ID", "time_window"], how="left")

sw = 1
df["Needed"] = 1
save_params = [
    "ID", "Age_first", "Gender_first", "Height_first", "BodyMass_first",
    "time_window", "window_start", "window_end",
]
cols_to_nan = [col for col in df.columns if col not in save_params]
result = []
for id_, grp in df.groupby("ID"):
    start = grp["time_window"].min()
    finish = grp["time_window"].max()
    while start <= finish:
        sw_end = start + sw
        mask = (grp["time_window"] >= start) & (grp["time_window"] < sw_end)
        searched_vals = grp[mask]
        hr_vals = searched_vals["HeartRate_median"]
        if hr_vals.isna().sum() > len(hr_vals) / 2:
            grp.loc[mask, cols_to_nan] = pd.NA
            grp.loc[mask, "Needed"] = 0
        else:
            grp.loc[mask, "Needed"] = 1
        start = sw_end
    result.append(grp)
df = pd.concat(result).sort_index()

eventbased_cols = [
    "AppleStandHour_sum", "AppleStandTime_sum", "AppleExerciseTime_sum",
    "DistanceWalkingRunning_sum", "StepCount_sum", "FlightsClimbed_sum",
]
df[eventbased_cols] = df[eventbased_cols].fillna(0)

gait_cols = [
    "WalkingStepLength_median", "WalkingSpeed_median",
    "WalkingAsymmetryPercentage_median", "WalkingDoubleSupportPercentage_median",
    "WalkingHeartRateAverage_median", "StairAscentSpeed_median",
    "StairDescentSpeed_median", "SixMinuteWalkTestDistance_median",
]
df[gait_cols] = df.groupby("ID")[gait_cols].ffill(limit=3)
df[gait_cols] = df.groupby("ID")[gait_cols].transform(lambda x: x.fillna(x.mean()))
df[gait_cols] = df[gait_cols].fillna(0)

noteventbased_cols = [
    "HeartRate_max", "HeartRate_median", "HeartRate_min", "HeartRate_std",
    "OxygenSaturation_max", "OxygenSaturation_median", "OxygenSaturation_min",
    "HeartRateVariabilitySDNN_max", "HeartRateVariabilitySDNN_median",
    "HeartRateVariabilitySDNN_min", "RestingHeartRate_median",
    "ActiveEnergyBurned_max", "ActiveEnergyBurned_median",
    "ActiveEnergyBurned_min", "ActiveEnergyBurned_sum",
    "BasalEnergyBurned_max", "BasalEnergyBurned_median",
    "BasalEnergyBurned_min", "BasalEnergyBurned_sum",
    "PhysicalEffort_max", "PhysicalEffort_median", "PhysicalEffort_min",
    "PhysicalEffort_sum",
]
df[noteventbased_cols] = df.groupby("ID")[noteventbased_cols].ffill(limit=3)
df[noteventbased_cols] = df.groupby("ID")[noteventbased_cols].transform(
    lambda x: x.fillna(x.mean()))
df[noteventbased_cols] = df[noteventbased_cols].fillna(df[noteventbased_cols].mean())

constant_cols = ["Age_first", "Gender_first", "Height_first", "BodyMass_first"]
df[constant_cols] = df.groupby("ID")[constant_cols].ffill().bfill()

for ele in ["HeartRate_median", "OxygenSaturation_median"]:
    df[f"diff_{ele}"] = df.groupby("ID")[ele].diff().fillna(0)
noteventbased_cols += ["diff_HeartRate_median", "diff_OxygenSaturation_median"]

df = df[df["Needed"] != 0].reset_index(drop=True)
df = df.drop(columns=["Needed"])

lags = [1, 3, 6]
diff_cols = [
    "HeartRate_median", "OxygenSaturation_median",
    "HeartRateVariabilitySDNN_median", "RestingHeartRate_median",
]
sum_cols = ["ActiveEnergyBurned_sum", "BasalEnergyBurned_sum", "PhysicalEffort_sum"]
df = df.sort_values(by=["ID", "time_window"])

for lag in lags:
    shifted = df.groupby("ID")[diff_cols].shift(lag)
    diff = df[diff_cols] - shifted
    diff.columns = [f"{col}_lag{lag}" for col in diff_cols]
    df = pd.concat([df, diff], axis=1)

shifted = df.groupby("ID")[sum_cols].shift(1)
rolling_sum = (
    shifted.groupby(df["ID"]).rolling(window=lags[2], min_periods=1).sum().droplevel(0))
rolling_sum.columns = [f"{col}_sum_lag" for col in sum_cols]
df = pd.concat([df, rolling_sum], axis=1)

lag_cols = [f"{col}_lag{lag}" for lag in lags for col in diff_cols] + [
    f"{col}_sum_lag" for col in sum_cols]
for col in lag_cols:
    df[col] = df.groupby("ID")[col].transform(lambda x: x.ffill().fillna(x.mean()))
    df[col] = df[col].fillna(df[col].mean())

df_base = df.copy()

# ═══════════════════════════════════════════════════════════════════════
# 2. LABEL DATA
# ═══════════════════════════════════════════════════════════════════════
df_label = pd.read_csv("export.csv")

for key, grp in df_label.groupby("study_id"):
    numeric_cols = grp.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        q1, q3 = grp[col].quantile(0.25), grp[col].quantile(0.75)
        iqr = q3 - q1
        df_label.loc[grp.index, f"{col}_outlier"] = ~grp[col].between(q1 - 1.5*iqr, q3 + 1.5*iqr)

df_label.loc[
    (df_label.index == 60) & (df_label["study_id"] == 5), "pulse_1_min"] = np.nan
df_label = df_label.drop(columns=[c for c in df_label.columns if c.endswith("_outlier")])

unnecessary_columns = [
    "redcap_event_name", "redcap_repeat_instrument", "redcap_repeat_instance",
    "date_enrolled", "dob", "comments", "demographics_complete", "ae_dateevent",
    "ae_description", "notes_define_other_event", "ae_unexpected",
    "ae_unexpectedcomment", "ae_related", "ae_reschrisk", "ae_serious",
    "ae_reportable", "ae_justification", "ae_completedby",
    "adverse_event_complete", "medication", "icd10_diagnosis",
    "wound_documentation", "fall_documentation", "mobilization_plan",
    "fluid_balance_intake", "fever", "caritas_protocol_information_complete",
    "dehydration_assessment_complete", "unintentional_weight_loss",
    "in_the_past_month_on_the_a", "if_yes_have_you_been_feeli", "weak",
    "frequency_weak", "using_the_scale_below_plea", "low_activity_level",
    "weakness", "frailtyassessment_complete", "vas_complete", "study_comments",
    "complete_study", "withdraw_date", "date_visit_4", "withdraw_reason",
    "completion_data_complete", "promis_pac_m_009r1_0dfa17",
    "promis_pac_m_105r1_2d14a6", "promis_pac_m_002r1_f9884e",
    "promis_pac_m_008r1_f1fe3b",
    "promis_ped_sf_v10_physical_activity_4a_no_survey_complete",
    "walk_time", "walk_speed",
]
df_label = df_label.drop(columns=[c for c in unnecessary_columns if c in df_label.columns])

df_label[["age", "gender", "height", "weight", "bmi"]] = (
    df_label.groupby("study_id")[["age", "gender", "height", "weight", "bmi"]]
    .ffill().bfill())

df_label = df_label[df_label.groupby("study_id").cumcount() >= 2].reset_index(drop=True)
df_label.loc[1, "date"] = "2024-02-21"
df_label.loc[134, "date"] = "2024-10-18"

frequ_param = [
    "pulse_1_min", "blood_pressure_mmhg", "blood_pressure_diastolic_m",
    "diarrhea", "dry_muscous_membranes", "is_the_skin_turgor_normal",
    "pain_medication", "self_reported_thirst", "capillary_refill_time", "vas_score",
]

df_label_base = df_label.copy()


# ═══════════════════════════════════════════════════════════════════════
# 3. PREPARE PAIN DATA
# ═══════════════════════════════════════════════════════════════════════
def prepare_pain_data():
    df = df_base.copy()
    dl = df_label_base.copy()

    fill_frequ = (
        dl.groupby("study_id")[frequ_param]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
        .reset_index())
    for col in frequ_param:
        dl[col] = dl[col].fillna(dl["study_id"].map(fill_frequ.set_index("study_id")[col]))

    dl["pain_label"] = (dl["vas_score"] > 4).astype(int)
    dl = dl.rename(columns={"study_id": "ID"})

    df["date"] = df["window_start"].dt.date
    dl["date"] = pd.to_datetime(dl["date"]).dt.date
    df = df.merge(dl[["ID", "date", "pain_label"]], on=["ID", "date"], how="left")
    df = df.drop(columns=["date"])
    df = df.dropna(subset=["pain_label"]).reset_index(drop=True)
    df = df.drop(["time_window", "window_start", "window_end"], axis=1)
    df = df.drop(["Height_first", "BodyMass_first"], axis=1, errors="ignore")

    measurement_counts = df.groupby("ID").size()
    quality_threshold = measurement_counts.median() * 0.3
    low_quality_ids = measurement_counts[measurement_counts < quality_threshold].index.tolist()
    df_pain = df[~df["ID"].isin(low_quality_ids)].copy()
    return df_pain


# ═══════════════════════════════════════════════════════════════════════
# 4. PREPARE DEHYDRATION DATA
# ═══════════════════════════════════════════════════════════════════════
def prepare_dehydration_data():
    df = df_base.copy()
    dl = df_label_base.copy()

    fill_frequ = (
        dl.groupby("study_id")[frequ_param]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
        .reset_index())
    for col in frequ_param:
        dl[col] = dl[col].fillna(dl["study_id"].map(fill_frequ.set_index("study_id")[col]))

    dl_calc = dl.copy()
    dl_calc["vc_mean"] = dl_calc.groupby("study_id")["vena_cava_diameter_mm"].transform("mean")

    raw_scores = []
    for _, row in dl_calc.iterrows():
        raw_score = 0
        if pd.notna(row["capillary_refill_time"]) and row["capillary_refill_time"] >= 2:
            raw_score += 5
        if pd.notna(row["is_the_skin_turgor_normal"]) and row["is_the_skin_turgor_normal"] != 1:
            raw_score += 5
        if pd.notna(row["diarrhea"]) and row["diarrhea"] != 0:
            raw_score += 5
        if pd.notna(row["blood_pressure_mmhg"]):
            if row["blood_pressure_mmhg"] <= 90: raw_score += 3
            elif row["blood_pressure_mmhg"] <= 100: raw_score += 2
            elif row["blood_pressure_mmhg"] <= 110: raw_score += 1
        if pd.notna(row["pulse_1_min"]):
            if row["pulse_1_min"] > 131: raw_score += 3
            elif row["pulse_1_min"] > 111: raw_score += 2
            elif row["pulse_1_min"] > 91: raw_score += 1
        if pd.notna(row["dry_muscous_membranes"]) and row["dry_muscous_membranes"] != 0:
            raw_score += 5
        if pd.notna(row["self_reported_thirst"]) and row["self_reported_thirst"] != 0:
            raw_score += 5
        if (pd.notna(row["vena_cava_diameter_mm"]) and pd.notna(row["vc_mean"])
                and row["vena_cava_diameter_mm"] < row["vc_mean"]):
            raw_score += 5
        raw_scores.append(raw_score)

    dl["dehydration_score_raw"] = pd.Series(raw_scores, index=dl.index)
    dl["dehydration_label"] = (dl["dehydration_score_raw"] > 10).astype(int)
    dl = dl.rename(columns={"study_id": "ID"})

    df["date"] = df["window_start"].dt.date
    dl["date"] = pd.to_datetime(dl["date"]).dt.date
    df = df.merge(dl[["ID", "date", "dehydration_label"]], on=["ID", "date"], how="left")
    df = df.drop(columns=["date"])
    df = df.dropna(subset=["dehydration_label"]).reset_index(drop=True)
    df = df.drop(["time_window", "window_start", "window_end"], axis=1)
    df = df.drop(["Height_first", "BodyMass_first"], axis=1, errors="ignore")

    measurement_counts = df.groupby("ID").size()
    quality_threshold = measurement_counts.median() * 0.3
    low_quality_ids = measurement_counts[measurement_counts < quality_threshold].index.tolist()
    df_dehy = df[~df["ID"].isin(low_quality_ids)].copy()
    return df_dehy


# ═══════════════════════════════════════════════════════════════════════
# 5. LOPOCV + FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════
def run_lopo_and_figures(df_data, label_col, params_file, label_name, line_color,
                         fig_avg_name, fig_supp_name, fig_shap_name):
    """Run full LOPOCV, generate averaged ROC, best-fold supplement, and SHAP."""

    with open(params_file) as f:
        best_params = json.load(f)

    label_std = df_data.groupby("ID")[label_col].std()
    label_changing_ids = sorted(label_std[label_std > 0].index.tolist())
    all_ids = sorted(df_data["ID"].unique().tolist())
    print(f"  Label-changing: {label_changing_ids} ({len(label_changing_ids)} folds)")
    print(f"  Total samples: {len(df_data)}")

    mean_fpr_grid = np.linspace(0, 1, 200)
    all_tprs = []
    best_auc = -1
    best_fold_data = None
    MIN_SHAP_SAMPLES = 80  # prefer folds with enough samples for visible SHAP

    for fold_idx, test_patient in enumerate(label_changing_ids):
        train_ids = [x for x in all_ids if x != test_patient]
        test_set = df_data[df_data["ID"] == test_patient]
        train_set = df_data[df_data["ID"].isin(train_ids)]

        X_tr = train_set.drop(columns=[label_col, "ID"])
        y_tr = train_set[label_col]
        X_te = test_set.drop(columns=[label_col, "ID"])
        y_te = test_set[label_col]

        if y_te.nunique() < 2:
            print(f"    Fold {fold_idx+1}: SKIP Patient {test_patient} (single class)")
            continue

        model = build_pipeline(y_train=y_tr, **best_params)
        model.fit(X_tr, y_tr)

        y_proba = model.predict_proba(X_te)[:, 1]
        pat_auc = roc_auc_score(y_te, y_proba)

        fpr_s, tpr_s, _ = roc_curve(y_te, y_proba)
        interp_tpr = np.interp(mean_fpr_grid, fpr_s, tpr_s)
        interp_tpr[0] = 0.0
        all_tprs.append(interp_tpr)

        # Prefer folds with enough test samples for a meaningful SHAP plot,
        # among those pick the highest AUC.
        n_te = len(y_te)
        if best_fold_data is None:
            pick = True
        elif n_te >= MIN_SHAP_SAMPLES and pat_auc > best_auc:
            pick = True
        elif n_te >= MIN_SHAP_SAMPLES and best_fold_data[3].shape[0] < MIN_SHAP_SAMPLES:
            pick = True  # first fold meeting the sample threshold
        else:
            pick = False

        if pick:
            best_auc = pat_auc
            best_fold_data = (model, X_tr.copy(), y_tr.copy(),
                              X_te.copy(), y_te.copy())

        print(f"    Fold {fold_idx+1}/{len(label_changing_ids)}: "
              f"Patient {test_patient} → AUC={pat_auc:.3f}")

    n_folds = len(all_tprs)

    # Averaged ROC
    print(f"\n  Generating {fig_avg_name} …")
    plot_averaged_roc(
        mean_fpr_grid, all_tprs,
        title=f"ROC — {label_name.capitalize()} (Averaged {n_folds} LOPO Folds)",
        fig_name=fig_avg_name,
        line_color=line_color,
        n_folds_label=f"Averaged across {n_folds} LOPO folds",
    )

    # Supplement best-fold ROC
    if best_fold_data is not None:
        bmodel, bX_tr, by_tr, bX_te, by_te = best_fold_data
        print(f"\n  Generating {fig_supp_name} (best fold AUC={best_auc:.3f}) …")
        plot_best_fold_roc(
            bmodel, bX_tr, by_tr, bX_te, by_te,
            title=f"ROC — {label_name.capitalize()} (Best LOPO Fold)",
            fig_name=fig_supp_name,
        )

        # SHAP
        print(f"\n  Generating {fig_shap_name} …")
        plot_shap_beeswarm(
            bmodel, bX_te,
            title=f"SHAP Feature Importance — {label_name.capitalize()}",
            fig_name=fig_shap_name,
        )

    return all_tprs, mean_fpr_grid


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PAIN")
print("=" * 60)
df_pain = prepare_pain_data()

pain_tprs, pain_fpr = run_lopo_and_figures(
    df_pain, "pain_label",
    "Pain_V7_Results/best_params_xgb_500.json",
    "Pain", "#1976D2",
    "Fig3_ROC_Pain",
    "Supplement_FigS1_ROC_Pain_BestSplit",
    "Fig5_SHAP_Pain",
)

print("\n" + "=" * 60)
print("DEHYDRATION")
print("=" * 60)
df_dehy = prepare_dehydration_data()

dehy_tprs, dehy_fpr = run_lopo_and_figures(
    df_dehy, "dehydration_label",
    "Dehydration_V7_Results/best_params_xgb_500.json",
    "Dehydration", "#D32F2F",
    "Fig4_ROC_Dehydration",
    "Supplement_FigS2_ROC_Dehydration_BestSplit",
    "Fig6_SHAP_Dehydration",
)

# ═══════════════════════════════════════════════════════════════════════
# 7. FEMALE-ONLY PAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PAIN — FEMALE ONLY")
print("=" * 60)

with open("Pain_V7_Results/best_params_xgb_500.json") as f:
    best_pain_params = json.load(f)

df_female = df_pain[df_pain["Gender_first"] == 0].copy()
female_ids = sorted(df_female["ID"].unique().tolist())
female_label_std = df_female.groupby("ID")["pain_label"].std().fillna(0)
female_changing_ids = sorted(female_label_std[female_label_std > 0].index.tolist())
print(f"  Female patients: {female_ids} ({len(female_ids)})")
print(f"  Label-changing: {female_changing_ids} ({len(female_changing_ids)})")

mean_fpr_grid = np.linspace(0, 1, 200)
female_tprs = []

for fold_idx, test_patient in enumerate(female_changing_ids):
    train_ids = [pid for pid in female_ids if pid != test_patient]
    train_set = df_female[df_female["ID"].isin(train_ids)]
    test_set = df_female[df_female["ID"] == test_patient]

    if (train_set.empty or test_set["pain_label"].nunique() < 2
            or train_set["pain_label"].nunique() < 2):
        print(f"    Skipping patient {test_patient}: insufficient class variation")
        continue

    X_tr = train_set.drop(columns=["pain_label", "ID"])
    y_tr = train_set["pain_label"]
    X_te = test_set.drop(columns=["pain_label", "ID"])
    y_te = test_set["pain_label"]

    model = build_pipeline(y_train=y_tr, **best_pain_params)
    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    pat_auc = roc_auc_score(y_te, y_proba)

    fpr_s, tpr_s, _ = roc_curve(y_te, y_proba)
    interp_tpr = np.interp(mean_fpr_grid, fpr_s, tpr_s)
    interp_tpr[0] = 0.0
    female_tprs.append(interp_tpr)
    print(f"    Fold {fold_idx+1}/{len(female_changing_ids)}: "
          f"Patient {test_patient} → AUC={pat_auc:.3f}")

if female_tprs:
    print(f"\n  Generating Fig_ROC_Pain_Female …")
    plot_averaged_roc(
        mean_fpr_grid, female_tprs,
        title="ROC — Pain, Female Only (Averaged 7 LOPO Folds)",
        fig_name="Fig_ROC_Pain_Female",
        line_color="#9C27B0",
        n_folds_label=f"Averaged across {len(female_tprs)} female LOPO folds",
    )

print("\n" + "=" * 60)
print("DONE — all figures in Publication_Figures/")
print("=" * 60)
