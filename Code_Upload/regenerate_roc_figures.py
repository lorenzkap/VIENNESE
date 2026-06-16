"""
Standalone script to regenerate the Publication ROC figures (Fig3 & Fig4)
by re-running the V7 LOPOCV pipeline and producing the averaged held-out
ROC curves with the correct AUC values in the legend.

Reproduces the exact same data pipeline as TexHype_V7.ipynb.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc as sklearn_auc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

PUB_DIR = "Publication_Figures"
os.makedirs(PUB_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING (mirrors notebook cells 8–30)
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
    columns="Name",
    values="Value",
    aggfunc="mean",
).reset_index()
df = all_data.merge(personal_data[["ID", "Gender", "Age"]], on="ID", how="left")

df.columns = df.columns.str.replace("HKCategoryTypeIdentifier", "", regex=True)
df.columns = df.columns.str.replace("HKQuantityTypeIdentifier", "", regex=True)
df[["Height", "BodyMass"]] = df.groupby("ID")[["Height", "BodyMass"]].ffill().bfill()

# Time align
date_cols = ["Start", "End", "enrollment_date", "final_visit"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
mask = (df["Start"].dt.date >= df["enrollment_date"].dt.date) & (
    df["End"].dt.date <= df["final_visit"].dt.date
)
df = df.loc[mask].copy()
df = df.drop(columns=["enrollment_date", "final_visit"]).reset_index(drop=True)

# Time windows
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

# Aggregate
group = df.groupby(["ID", "time_window"])
df = group.agg(
    {
        "Age": "first",
        "Gender": "first",
        "Height": "first",
        "BodyMass": "first",
        "HeartRate": ["max", "median", "min", "std"],
        "OxygenSaturation": ["max", "median", "min"],
        "HeartRateVariabilitySDNN": ["max", "median", "min"],
        "RestingHeartRate": ["median"],
        "ActiveEnergyBurned": ["max", "median", "min", "sum"],
        "BasalEnergyBurned": ["max", "median", "min", "sum"],
        "PhysicalEffort": ["max", "median", "min", "sum"],
        "AppleStandHour": ["sum"],
        "AppleStandTime": ["sum"],
        "AppleExerciseTime": ["sum"],
        "DistanceWalkingRunning": ["sum"],
        "StepCount": ["sum"],
        "WalkingStepLength": ["median"],
        "WalkingSpeed": ["median"],
        "WalkingAsymmetryPercentage": ["median"],
        "WalkingDoubleSupportPercentage": ["median"],
        "WalkingHeartRateAverage": ["median"],
        "FlightsClimbed": ["sum"],
        "StairAscentSpeed": ["median"],
        "StairDescentSpeed": ["median"],
        "SixMinuteWalkTestDistance": ["median"],
    }
).reset_index()

df.columns = [
    col[0] if isinstance(col, tuple) and col[1] == "" else f"{col[0]}_{col[1]}"
    for col in df.columns
]
df = pd.merge(tw_df, df, on=["ID", "time_window"], how="left")

# Cut inside
sw = 1
df["Needed"] = 1
save_params = [
    "ID", "Age_first", "Gender_first", "Height_first", "BodyMass_first",
    "time_window", "window_start", "window_end",
]
cols_to_nan = [col for col in df.columns if col not in save_params]
result = []
for id_, group in df.groupby("ID"):
    start = group["time_window"].min()
    finish = group["time_window"].max()
    while start <= finish:
        sw_end = start + sw
        mask = (group["time_window"] >= start) & (group["time_window"] < sw_end)
        searched_vals = group[mask]
        hr_vals = searched_vals["HeartRate_median"]
        if hr_vals.isna().sum() > len(hr_vals) / 2:
            group.loc[mask, cols_to_nan] = pd.NA
            group.loc[mask, "Needed"] = 0
        else:
            group.loc[mask, "Needed"] = 1
        start = sw_end
    result.append(group)
df = pd.concat(result).sort_index()

# Impute
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
    lambda x: x.fillna(x.mean())
)
df[noteventbased_cols] = df[noteventbased_cols].fillna(df[noteventbased_cols].mean())

constant_cols = ["Age_first", "Gender_first", "Height_first", "BodyMass_first"]
df[constant_cols] = df.groupby("ID")[constant_cols].ffill().bfill()

for ele in ["HeartRate_median", "OxygenSaturation_median"]:
    df[f"diff_{ele}"] = df.groupby(["ID"])[ele].diff().fillna(0)
noteventbased_cols = noteventbased_cols + [
    "diff_HeartRate_median", "diff_OxygenSaturation_median",
]

# Clean
df = df[df["Needed"] != 0].reset_index(drop=True)
df = df.drop(columns=["Needed"])

# Lags
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
    shifted.groupby(df["ID"]).rolling(window=lags[2], min_periods=1).sum().droplevel(0)
)
rolling_sum.columns = [f"{col}_sum_lag" for col in sum_cols]
df = pd.concat([df, rolling_sum], axis=1)

lag_cols = [f"{col}_lag{lag}" for lag in lags for col in diff_cols] + [
    f"{col}_sum_lag" for col in sum_cols
]
for col in lag_cols:
    df[col] = df.groupby("ID")[col].transform(lambda x: x.ffill().fillna(x.mean()))
    df[col] = df[col].fillna(df[col].mean())

# Save base state (before pain/dehydration label merging)
df_base = df.copy()

# ═══════════════════════════════════════════════════════════════════════
# 2. LABEL DATA
# ═══════════════════════════════════════════════════════════════════════
df_label = pd.read_csv("export.csv")

# Outlier detection
for key, group in df_label.groupby("study_id"):
    numeric_cols = group.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        q1 = group[col].quantile(0.25)
        q3 = group[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_label.loc[group.index, f"{col}_outlier"] = ~group[col].between(lower, upper)

df_label.loc[
    (df_label.index == 60) & (df_label["study_id"] == 5), "pulse_1_min"
] = np.nan

df_label = df_label.drop(
    columns=[col for col in df_label.columns if col.endswith("_outlier")]
)

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
    .ffill()
    .bfill()
)

# Drop enrollment/final-visit rows
df_label = df_label[df_label.groupby("study_id").cumcount() >= 2].reset_index(
    drop=True
)
df_label.loc[1, "date"] = "2024-02-21"
df_label.loc[134, "date"] = "2024-10-18"

df_label_base = df_label.copy()


# ═══════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
def build_pipeline(**xgb_params):
    # Filter out non-XGBoost params (original dehydration params were from RF)
    valid_xgb = {
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma",
        "reg_alpha", "reg_lambda", "max_leaves", "grow_policy",
        "scale_pos_weight", "base_score", "booster", "n_jobs",
    }
    filtered = {k: v for k, v in xgb_params.items() if k in valid_xgb}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", XGBClassifier(random_state=42, eval_metric="logloss", **filtered)),
    ])


def save_avg_holdout_roc(mean_fpr, tpr_curves, label_name, output_dir, fig_name,
                         holdout_label, line_color):
    tprs = np.asarray(tpr_curves)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    auc_values = np.array([sklearn_auc(mean_fpr, tpr_i) for tpr_i in tprs])
    mean_auc = sklearn_auc(mean_fpr, mean_tpr)
    std_auc = auc_values.std()
    std_tpr = tprs.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    for tpr_i in tprs:
        ax.plot(mean_fpr, tpr_i, alpha=0.12, linewidth=0.8, color=line_color)

    ax.plot(
        mean_fpr, mean_tpr, color=line_color, linewidth=2.2,
        label=f"Mean held-out ROC (AUC = {mean_auc:.2f} \u00b1 {std_auc:.2f})",
    )
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.18, color=line_color, label="\u00b1 1 SD",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Chance")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curve - {label_name.capitalize()} Estimation")
    ax.text(
        0.04, 0.06, holdout_label, transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9),
    )
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_aspect("equal")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, fig_name + ".tif"), format="tiff", dpi=300)
    plt.savefig(os.path.join(output_dir, fig_name + ".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved {fig_name}  (AUC = {mean_auc:.4f} \u00b1 {std_auc:.4f})")
    return mean_auc, std_auc


# ═══════════════════════════════════════════════════════════════════════
# 4. PAIN — impute labels, merge, LOPO, figure
# ═══════════════════════════════════════════════════════════════════════
print("\n=== PAIN ===")

df = df_base.copy()
df_label = df_label_base.copy()

# Impute label data
frequ_param = [
    "pulse_1_min", "blood_pressure_mmhg", "blood_pressure_diastolic_m",
    "diarrhea", "dry_muscous_membranes", "is_the_skin_turgor_normal",
    "pain_medication", "self_reported_thirst", "capillary_refill_time", "vas_score",
]
fill_frequ = (
    df_label.groupby("study_id")[frequ_param]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
    .reset_index()
)
for col in frequ_param:
    df_label[col] = df_label[col].fillna(
        df_label["study_id"].map(fill_frequ.set_index("study_id")[col])
    )

# Pain label: VAS > 4
df_label["pain_label"] = (df_label["vas_score"] > 4).astype(int)
df_label = df_label.rename(columns={"study_id": "ID"})

df["date"] = df["window_start"].dt.date
df_label["date"] = pd.to_datetime(df_label["date"]).dt.date
df = df.merge(df_label[["ID", "date", "pain_label"]], on=["ID", "date"], how="left")
df = df.drop(columns=["date"])
df = df.dropna(subset=["pain_label"]).reset_index(drop=True)
df = df.drop(["time_window", "window_start", "window_end"], axis=1)
df = df.drop(["Height_first", "BodyMass_first"], axis=1, errors="ignore")

# Quality filter
measurement_counts = df.groupby("ID").size()
median_count = measurement_counts.median()
quality_threshold = median_count * 0.3
low_quality_ids = measurement_counts[measurement_counts < quality_threshold].index.tolist()
df_pain = df[~df["ID"].isin(low_quality_ids)].copy()

label_std = df_pain.groupby("ID")["pain_label"].std()
label_changing_ids = sorted(label_std[label_std > 0].index.tolist())
all_ids = sorted(df_pain.ID.unique().tolist())
print(f"  Label-changing patients: {label_changing_ids} ({len(label_changing_ids)})")
print(f"  Total samples: {len(df_pain)}")

# Load best params
with open("Pain_V7_Results/best_params_xgb_500.json") as f_:
    best_pain_params = json.load(f_)

# LOPO
mean_fpr_grid = np.linspace(0, 1, 200)
all_pain_test_tprs = []

for fold_idx, test_patient in enumerate(label_changing_ids):
    train_ids = [x for x in all_ids if x != test_patient]
    test_set = df_pain[df_pain["ID"] == test_patient]
    train_set = df_pain[df_pain["ID"].isin(train_ids)]

    X_tr = train_set.drop(columns=["pain_label", "ID"])
    y_tr = train_set["pain_label"]
    X_te = test_set.drop(columns=["pain_label", "ID"])
    y_te = test_set["pain_label"]

    model = build_pipeline(**best_pain_params)
    model.fit(X_tr, y_tr)

    y_proba = model.predict_proba(X_te)[:, 1]
    pat_auc = roc_auc_score(y_te, y_proba)

    fpr_s, tpr_s, _ = roc_curve(y_te, y_proba)
    interp_tpr = np.interp(mean_fpr_grid, fpr_s, tpr_s)
    interp_tpr[0] = 0.0
    all_pain_test_tprs.append(interp_tpr)

    print(f"  Fold {fold_idx+1}/{len(label_changing_ids)}: Patient {test_patient} → AUC={pat_auc:.3f}")

save_avg_holdout_roc(
    mean_fpr_grid, all_pain_test_tprs, "pain", PUB_DIR, "Fig3_ROC_Pain",
    f"Averaged held-out ROC across {len(label_changing_ids)} LOPO folds", "#1976D2",
)

# ═══════════════════════════════════════════════════════════════════════
# 5. DEHYDRATION — impute labels, merge, LOPO, figure
# ═══════════════════════════════════════════════════════════════════════
print("\n=== DEHYDRATION ===")

df = df_base.copy()
df_label = df_label_base.copy()

# Impute label data
fill_frequ = (
    df_label.groupby("study_id")[frequ_param]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
    .reset_index()
)
for col in frequ_param:
    df_label[col] = df_label[col].fillna(
        df_label["study_id"].map(fill_frequ.set_index("study_id")[col])
    )

# Dehydration label: raw score > 10
df_label_calc = df_label.copy()
df_label_calc["vc_mean"] = df_label_calc.groupby("study_id")[
    "vena_cava_diameter_mm"
].transform("mean")

raw_scores = []
for _, row in df_label_calc.iterrows():
    raw_score = 0
    if pd.notna(row["capillary_refill_time"]) and row["capillary_refill_time"] >= 2:
        raw_score += 5
    if pd.notna(row["is_the_skin_turgor_normal"]) and row["is_the_skin_turgor_normal"] != 1:
        raw_score += 5
    if pd.notna(row["diarrhea"]) and row["diarrhea"] != 0:
        raw_score += 5
    if pd.notna(row["blood_pressure_mmhg"]):
        if row["blood_pressure_mmhg"] <= 90:
            raw_score += 3
        elif row["blood_pressure_mmhg"] <= 100:
            raw_score += 2
        elif row["blood_pressure_mmhg"] <= 110:
            raw_score += 1
    if pd.notna(row["pulse_1_min"]):
        if row["pulse_1_min"] > 131:
            raw_score += 3
        elif row["pulse_1_min"] > 111:
            raw_score += 2
        elif row["pulse_1_min"] > 91:
            raw_score += 1
    if pd.notna(row["dry_muscous_membranes"]) and row["dry_muscous_membranes"] != 0:
        raw_score += 5
    if pd.notna(row["self_reported_thirst"]) and row["self_reported_thirst"] != 0:
        raw_score += 5
    if (
        pd.notna(row["vena_cava_diameter_mm"])
        and pd.notna(row["vc_mean"])
        and row["vena_cava_diameter_mm"] < row["vc_mean"]
    ):
        raw_score += 5
    raw_scores.append(raw_score)

df_label["dehydration_score_raw"] = pd.Series(raw_scores, index=df_label.index)
df_label["dehydration_label"] = (df_label["dehydration_score_raw"] > 10).astype(int)
df_label = df_label.rename(columns={"study_id": "ID"})

df["date"] = df["window_start"].dt.date
df_label["date"] = pd.to_datetime(df_label["date"]).dt.date
df = df.merge(
    df_label[["ID", "date", "dehydration_label"]], on=["ID", "date"], how="left"
)
df = df.drop(columns=["date"])
df = df.dropna(subset=["dehydration_label"]).reset_index(drop=True)
df = df.drop(["time_window", "window_start", "window_end"], axis=1)
df = df.drop(["Height_first", "BodyMass_first"], axis=1, errors="ignore")

# Quality filter
measurement_counts = df.groupby("ID").size()
median_count = measurement_counts.median()
quality_threshold = median_count * 0.3
low_quality_ids = measurement_counts[measurement_counts < quality_threshold].index.tolist()
df_dehy = df[~df["ID"].isin(low_quality_ids)].copy()

label_std = df_dehy.groupby("ID")["dehydration_label"].std()
label_changing_ids = sorted(label_std[label_std > 0].index.tolist())
all_ids = sorted(df_dehy.ID.unique().tolist())
print(f"  Label-changing patients: {label_changing_ids} ({len(label_changing_ids)})")
print(f"  Total samples: {len(df_dehy)}")

# Load best params
with open("Dehydration_V7_Results/best_params_xgb_500.json") as f_:
    best_dehy_params = json.load(f_)

# LOPO
all_dehy_test_tprs = []

for fold_idx, test_patient in enumerate(label_changing_ids):
    train_ids = [x for x in all_ids if x != test_patient]
    test_set = df_dehy[df_dehy["ID"] == test_patient]
    train_set = df_dehy[df_dehy["ID"].isin(train_ids)]

    X_tr = train_set.drop(columns=["dehydration_label", "ID"])
    y_tr = train_set["dehydration_label"]
    X_te = test_set.drop(columns=["dehydration_label", "ID"])
    y_te = test_set["dehydration_label"]

    if y_te.nunique() < 2:
        print(f"  Fold {fold_idx+1}: SKIP Patient {test_patient} (single class)")
        continue

    model = build_pipeline(**best_dehy_params)
    model.fit(X_tr, y_tr)

    y_proba = model.predict_proba(X_te)[:, 1]
    pat_auc = roc_auc_score(y_te, y_proba)

    fpr_s, tpr_s, _ = roc_curve(y_te, y_proba)
    interp_tpr = np.interp(mean_fpr_grid, fpr_s, tpr_s)
    interp_tpr[0] = 0.0
    all_dehy_test_tprs.append(interp_tpr)

    print(f"  Fold {fold_idx+1}/{len(label_changing_ids)}: Patient {test_patient} → AUC={pat_auc:.3f}")

n_evaluated = len(all_dehy_test_tprs)
save_avg_holdout_roc(
    mean_fpr_grid, all_dehy_test_tprs, "dehydration", PUB_DIR, "Fig4_ROC_Dehydration",
    f"Averaged held-out ROC across {n_evaluated} LOPO folds", "#D32F2F",
)


# ═══════════════════════════════════════════════════════════════════════
# 6. GENDER-STRATIFIED PAIN — female-only LOPOCV with ROC
# ═══════════════════════════════════════════════════════════════════════
print("\n=== PAIN — FEMALE-ONLY GENDER ANALYSIS ===")

df_female = df_pain[df_pain["Gender_first"] == 0].copy()
female_ids = sorted(df_female["ID"].unique().tolist())
female_label_std = df_female.groupby("ID")["pain_label"].std().fillna(0)
female_changing_ids = sorted(female_label_std[female_label_std > 0].index.tolist())

print(f"  Female patients: {female_ids} ({len(female_ids)})")
print(f"  Label-changing: {female_changing_ids} ({len(female_changing_ids)})")
print(f"  Total samples: {len(df_female)}")


def build_group_pain_pipeline(y_train, **xgb_params):
    """Same model but relax SMOTE k_neighbors for small subgroup folds."""
    valid_xgb = {
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma",
        "reg_alpha", "reg_lambda", "max_leaves", "grow_policy",
        "scale_pos_weight", "base_score", "booster", "n_jobs",
    }
    filtered = {k: v for k, v in xgb_params.items() if k in valid_xgb}

    class_counts = y_train.value_counts()
    minority_count = int(class_counts.min()) if not class_counts.empty else 0
    if minority_count >= 2:
        smote_step = SMOTE(random_state=42, k_neighbors=min(3, minority_count - 1))
    else:
        smote_step = "passthrough"

    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", smote_step),
        ("classifier", XGBClassifier(random_state=42, eval_metric="logloss", **filtered)),
    ])


all_female_tprs = []

for fold_idx, test_patient in enumerate(female_changing_ids):
    train_ids = [pid for pid in female_ids if pid != test_patient]
    train_set = df_female[df_female["ID"].isin(train_ids)]
    test_set = df_female[df_female["ID"] == test_patient]

    if train_set.empty or test_set["pain_label"].nunique() < 2 or train_set["pain_label"].nunique() < 2:
        print(f"  Skipping patient {test_patient}: insufficient class variation")
        continue

    X_tr = train_set.drop(columns=["pain_label", "ID"])
    y_tr = train_set["pain_label"]
    X_te = test_set.drop(columns=["pain_label", "ID"])
    y_te = test_set["pain_label"]

    model = build_group_pain_pipeline(y_tr, **best_pain_params)
    model.fit(X_tr, y_tr)

    y_proba = model.predict_proba(X_te)[:, 1]
    pat_auc = roc_auc_score(y_te, y_proba)

    fpr_s, tpr_s, _ = roc_curve(y_te, y_proba)
    interp_tpr = np.interp(mean_fpr_grid, fpr_s, tpr_s)
    interp_tpr[0] = 0.0
    all_female_tprs.append(interp_tpr)

    print(f"  Fold {fold_idx+1}/{len(female_changing_ids)}: Patient {test_patient} → AUC={pat_auc:.3f}")

if all_female_tprs:
    save_avg_holdout_roc(
        mean_fpr_grid, all_female_tprs, "pain (female only)", PUB_DIR,
        "Fig_ROC_Pain_Female",
        f"Averaged held-out ROC across {len(all_female_tprs)} female LOPO folds",
        "#9C27B0",
    )

print("\nDone. Publication ROC figures regenerated in Publication_Figures/")
