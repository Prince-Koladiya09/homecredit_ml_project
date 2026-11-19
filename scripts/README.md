
# Version 2.1 Experiment Runner

Run all commands from the project root.

## Main experiment

```powershell
python scripts/run_full_experiments.py
```

The script performs five-fold stratified cross-validation for:

- CatBoost depth 6
- CatBoost depth 7
- CatBoost depth 6 without the employment sentinel indicator
- LightGBM
- XGBoost

It saves out-of-fold and test predictions in:

```text
outputs/full_experiments/
```

The final blended submission is saved as:

```text
outputs/submission_v2_1.csv
```

## EDA

```powershell
python src/eda.py
```

Outputs are saved under:

```text
outputs/eda/
```

## Important notes

- Do not delete cached `.npy` predictions unless you want to rerun those models.
- Cache signatures are checked against the feature columns and model configuration.
- The blend weights are selected using OOF predictions. The blend score can be optimistic.
- Do not treat OOF AUC as the public leaderboard score.
- Keep competition data private and follow the competition's data-sharing rules.
- Check the submission ID order and sample submission format before submitting.
