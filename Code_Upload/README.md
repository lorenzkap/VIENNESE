# VIENNESE — Smartwatch-Based Pain and Dehydration Detection in Nursing Home Residents

Analysis code for the VIENNESE study (Vital sIgn wEarables NursiNg homE SEtting), a
prospective, single-center observational feasibility study evaluating continuous
smartwatch monitoring and patient-independent machine-learning models for pain and
dehydration detection in nursing home residents.

This repository accompanies the manuscript:

> Kapral L, Bucek F, Lichtenegger L, Albrecht A, Berger L, Moscato F, Willschke H.
> *Patient-Level Evaluation of Smartwatch-Based Pain and Dehydration Detection in
> Nursing Home Residents: A Prospective Observational Study.*

## What this code does

The pipeline aggregates Apple Watch data into 120-minute time windows, engineers 56
features across vital signs, activity/energy, gait/mobility, and demographics, applies
a patient-level data-quality filter, and trains matched XGBoost classifiers for two
binary endpoints:

- **Pain** — visual analog scale (VAS) > 4
- **Dehydration** — uncapped composite clinical raw score > 10

Models are evaluated with **Leave-One-Patient-Out Cross-Validation (LOPOCV)**. The code
reproduces every figure and the pooled / per-patient performance metrics reported in the
manuscript and the Multimedia Appendix.

## Repository contents

```
.
├── README.md
├── requirements.txt
├── regenerate_all_figures.py          # Main reproducible pipeline (LOPOCV + all figures)
├── regenerate_publication_figures.py  # Publication-styled figure variants
├── regenerate_roc_figures.py          # ROC-curve regeneration helpers
├── VIENNESE_analysis.ipynb            # Full exploratory analysis notebook (outputs stripped)
└── params/
    ├── pain_best_params_xgb_500.json         # Tuned XGBoost hyperparameters (pain)
    └── dehydration_best_params_xgb_500.json  # Tuned XGBoost hyperparameters (dehydration)
```

## Data availability

**Individual-level participant data are not included in this repository.** Because the
data come from nursing home residents and are highly sensitive (and in part directly
identifying, e.g., dates of birth), they cannot be shared publicly. Aggregate outputs may
be made available upon reasonable request to the corresponding author, subject to ethics
committee approval.

To run the pipeline you must provide the following input files in the project root (not
distributed here):

| File               | Description                                                            |
|--------------------|------------------------------------------------------------------------|
| `output_all.csv`   | Aggregated per-window Apple Watch features (one row per patient/window) |
| `personal_data.csv`| Per-patient demographics (`ID`, `Year of Birth`, `Sex`, …)             |
| `export.csv`       | Clinical assessment labels (pain VAS and dehydration sub-indicators)    |

The tuned hyperparameters in `params/` are loaded by the scripts. The scripts reference
them under `Pain_V7_Results/` and `Dehydration_V7_Results/`; create those folders (or
adjust the paths at the top of the script) and place the corresponding
`best_params_xgb_500.json` file in each.

## Environment

- Python 3.11
- Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the results

With the input data files in place:

```bash
python regenerate_all_figures.py
```

This runs the full LOPOCV for both endpoints (including the exploratory female-only pain
analysis) and writes all publication and supplementary figures to `Publication_Figures/`.

## Method summary

- **Windowing:** 120-minute non-overlapping windows aligned to clinical assessments;
  windows with HR coverage < 50% are dropped as non-wear.
- **Quality filter:** patients contributing fewer than 30% of the median sample count are
  excluded (resulting in 13 of 16 patients, 1414 windows).
- **Model:** `StandardScaler` → `XGBClassifier`, hyperparameters tuned with
  `RandomizedSearchCV` (500 iterations, 3-fold CV, AUC scoring) with one patient held out
  inside the tuning split to mirror the outer LOPOCV.
- **Evaluation:** pooled held-out metrics and mean held-out-patient AUC ± SD across folds;
  feature importance fold-averaged; SHAP beeswarm plots from the retained publication
  models.

## License

Released under the MIT License (see `LICENSE`).
