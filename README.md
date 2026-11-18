# Home Credit Default Risk — Competition Project

A reproducible tabular binary-classification project for the GCI World 2026 in-class Home Credit competition. The task is to predict the probability of payment difficulties (`TARGET=1`). The supplied course materials specify ROC-AUC as the evaluation metric.

## Competition rules reflected in this project

- Use only the supplied competition training and test data. Do not add external data.
- Generate predictions with a trained model; manual or hand-labelled predictions are not permitted.
- Record random seeds and retain the code/configuration that generated the submitted file.
- The submission must contain `SK_ID_CURR,TARGET` and preserve the sample submission's ID order.
- The supplied README states that the last uploaded submission determines the final ranking. Public leaderboard results are based on a subset; final scoring uses the private leaderboard.
- The course tutorial describes a reproducible code archive for Honors/Outstanding consideration. Check the current Omnicampus portal for the exact upload format and requirements.
- Cite code that is actually reused from external sources. Do not claim that code was independently written if it was copied or adapted.

## Project contents

```text
homecredit_ml_project/
├── input/                  # supplied data and course reference materials
├── src/
│   ├── eda.py              # reproducible EDA tables and chart generation
│   ├── features.py         # deterministic feature engineering
│   ├── inspect_data.py     # data schema/quality profile
│   └── train.py            # validation, full-data refit and submission
├── notebooks/
│   ├── 01_data_audit_eda.ipynb  # executed EDA notebook with chart outputs
│   └── README.md
├── docs/
│   ├── DEVELOPMENT_EVIDENCE.md  # staged workflow, decisions and experiment log
│   └── PROJECT_PLAN.md
├── outputs/
│   ├── eda/                # EDA charts, CSV tables and summary JSON
│   ├── submission.csv      # generated test probabilities
│   ├── validation_metrics.json
│   ├── validation_predictions.csv
│   ├── run_summary.json
│   └── data_profile.json
├── models/homecredit_model.joblib
├── requirements.txt
└── README.md
```

## Environment setup

Python 3.10 or newer is recommended. From the project root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the EDA

From the project root:

```bash
python src/inspect_data.py
python src/eda.py
```

The second command regenerates the figures and tables in `outputs/eda/`. The executed notebook `notebooks/01_data_audit_eda.ipynb` contains the charts and explains their interpretation. Open it from the project root so its relative paths resolve.

### Included EDA coverage

1. Target class balance.
2. Train/test missingness comparison.
3. Selected numerical feature distributions (including skewed amount variables).
4. `DAYS_EMPLOYED` sentinel-like value pattern.
5. Correlation heatmap for selected numerical variables.
6. Target rate by each categorical feature (categories with at least 30 rows; sample counts shown).
7. Numerical train/test shift using standardized mean differences.
8. Categorical train/test shift using total variation distance.
9. Train/test missingness-rate differences.
10. `EXT_SOURCE` distributions by target class.

The plots are descriptive, not causal. Small category groups can have unstable target rates. Train/test similarity checks cannot prove that distributions are identical or that test labels follow the same conditional distribution.

## Train and evaluate

The current training entry point supports three model families:

```bash
python src/train.py --model histgb --seed 42
python src/train.py --model random_forest --seed 42
python src/train.py --model extra_trees --seed 42
```

Each run uses a stratified 80/20 holdout with the requested seed, fits imputation and one-hot encoding within the pipeline, reports ROC-AUC plus average precision and Brier score, refits the model on all labelled rows, and writes a new test submission. **Each run overwrites the current output and model files.** Copy/archive each run's metrics and submission before running another configuration.

To run a final full-data fit after selecting a configuration from recorded validation experiments:

```bash
python src/train.py --model histgb --seed 42 --skip-validation
```

Use `--skip-validation` only after selecting the model/configuration from actual validation evidence. The saved HistGradientBoosting validation result currently included in this archive is a single stratified holdout result (seed 42): ROC-AUC **0.751440**, average precision **0.235125**, Brier score **0.068237**, on 34,241 validation rows. This is a local holdout score, not a competition leaderboard score and not evidence that HistGradientBoosting is superior to the alternatives.

## Submission validation

After training, run this check from the project root:

```bash
python - <<'PY'
import numpy as np
import pandas as pd
p = pd.read_csv('outputs/submission.csv')
s = pd.read_csv('input/sample_submission.csv')
t = pd.read_csv('input/test.csv')
assert p.shape == (61_500, 2)
assert p.columns.tolist() == ['SK_ID_CURR', 'TARGET']
assert p['SK_ID_CURR'].is_unique
assert p['SK_ID_CURR'].equals(s['SK_ID_CURR'])
assert p['SK_ID_CURR'].equals(t['SK_ID_CURR'])
assert p['TARGET'].notna().all()
assert np.isfinite(p['TARGET']).all()
assert p['TARGET'].between(0, 1).all()
print('Submission checks passed:', p.shape)
PY
```

Upload `outputs/submission.csv` through Omnicampus. Confirm on the portal that the intended file is the latest submission; retain the uploaded file and timestamp.

## Feature and preprocessing decisions

- `SK_ID_CURR` is retained for joining/submission, not used as a model predictor.
- Numeric missing values are median-imputed with missing indicators; categorical missing values use a dedicated token and one-hot encoding handles unseen categories.
- Existing category strings such as `XNA` are preserved as categories rather than automatically treated as nulls.
- The feature builder adds age, employment duration/sentinel flag, financial ratios and `EXT_SOURCE` summaries. These are hypotheses and should be retained only if fair comparisons show value.
- Positive `DAYS_EMPLOYED` values are flagged as sentinel-like and excluded from the derived employment-years calculation; the raw column remains available.

## Evidence and limitations — read before submission

`docs/DEVELOPMENT_EVIDENCE.md` records the staged workflow, the observed initial run and experiment-log templates. The archive contains a real EDA run and a HistGradientBoosting holdout run. It does **not** claim a completed, fair model-family/feature-ablation comparison or a hyperparameter search. Complete and record those comparisons if time permits; otherwise state this limitation honestly. Do not describe the included model as the best model unless the experiment log supports that decision.

A clean-environment end-to-end rerun and the current Omnicampus ZIP upload rules must be checked by the submitter. The presence of a generated submission in this archive does not prove that it has been uploaded or accepted by the competition platform.

## AI assistance and academic integrity

If AI tools were used for planning, code drafting, debugging, or review, disclose that assistance according to the course policy and retain the actual conversation evidence if required. Verify all generated code and results yourself. Do not fabricate experiments, scores, personal work, or AI conversations.
