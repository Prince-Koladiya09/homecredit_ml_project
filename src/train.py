"""Train/evaluate a reproducible Home Credit default model and create submission.csv."""
from __future__ import annotations
import argparse, json, logging, os, random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from features import HomeCreditFeatureBuilder

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("homecredit")

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)

def make_pipeline(X: pd.DataFrame, model_name="histgb", seed=SEED):
    feature_builder = HomeCreditFeatureBuilder()
    transformed = feature_builder.fit_transform(X)
    cat_cols = transformed.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in transformed.columns if c not in cat_cols]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric, num_cols),
        ("categorical", categorical, cat_cols),
    ], remainder="drop", verbose_feature_names_out=False)
    if model_name == "histgb":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=180, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=15,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        )
    elif model_name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=350, max_depth=16, min_samples_leaf=12,
            max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("features", feature_builder), ("preprocess", prep), ("model", estimator)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "input/train.csv"))
    parser.add_argument("--test", default=str(ROOT / "input/test.csv"))
    parser.add_argument("--sample", default=str(ROOT / "input/sample_submission.csv"))
    parser.add_argument("--model", choices=["histgb", "random_forest", "extra_trees"], default="histgb")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--skip-validation", action="store_true", help="Fit on all training rows; no holdout metric")
    args = parser.parse_args()
    seed_everything(args.seed)
    out = ROOT / "outputs"; models = ROOT / "models"
    out.mkdir(exist_ok=True); models.mkdir(exist_ok=True)
    train = pd.read_csv(args.train); test = pd.read_csv(args.test); sample = pd.read_csv(args.sample)
    if "TARGET" not in train: raise ValueError("train.csv must contain TARGET")
    if not sample["SK_ID_CURR"].equals(test["SK_ID_CURR"]):
        raise ValueError("sample_submission IDs/order do not match test.csv")
    X = train.drop(columns="TARGET"); y = train["TARGET"].astype(int)
    if not args.skip_validation:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=args.valid_size, stratify=y, random_state=args.seed
        )
        validation_model = make_pipeline(X_tr, args.model, args.seed)
        LOG.info("Fitting %s on %d rows; validating on %d rows", args.model, len(X_tr), len(X_va))
        validation_model.fit(X_tr, y_tr)
        valid_pred = validation_model.predict_proba(X_va)[:, 1]
        metrics = {
            "model": args.model, "seed": args.seed, "validation_fraction": args.valid_size,
            "validation_rows": int(len(X_va)), "roc_auc": float(roc_auc_score(y_va, valid_pred)),
            "average_precision": float(average_precision_score(y_va, valid_pred)),
            "brier_score": float(brier_score_loss(y_va, valid_pred)),
        }
        (out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
        pd.DataFrame({"y_true": y_va.to_numpy(), "prediction": valid_pred}).to_csv(out / "validation_predictions.csv", index=False)
        LOG.info("Validation ROC-AUC: %.6f | AP: %.6f", metrics["roc_auc"], metrics["average_precision"])
    else:
        metrics = {"model": args.model, "seed": args.seed, "note": "Validation skipped by request"}
    # Refit the chosen configuration on all labelled rows before predicting test.
    final_model = make_pipeline(X, args.model, args.seed)
    LOG.info("Refitting final model on all %d labelled rows", len(X))
    final_model.fit(X, y)
    pred = final_model.predict_proba(test)[:, 1]
    if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    submission = sample.copy(); submission["TARGET"] = pred
    submission.to_csv(out / "submission.csv", index=False)
    joblib.dump(final_model, models / "homecredit_model.joblib")
    run_info = {**metrics, "train_rows": len(train), "test_rows": len(test), "features_before_engineering": X.shape[1],
                "prediction_min": float(pred.min()), "prediction_max": float(pred.max()),
                "submission_path": "outputs/submission.csv"}
    (out / "run_summary.json").write_text(json.dumps(run_info, indent=2))
    LOG.info("Saved %s; prediction range [%.5f, %.5f]", out / "submission.csv", pred.min(), pred.max())

if __name__ == "__main__": main()
