
"""
Home Credit Version 2.1 experiment runner.

Run from the project root:
    python scripts/run_full_experiments.py

Models:
    CatBoost depth 6
    CatBoost depth 7
    CatBoost depth 6 without employment sentinel
    LightGBM
    XGBoost

Outputs:
    outputs/full_experiments/
    outputs/submission_v2_1.csv
"""

from __future__ import annotations

import gc
import hashlib
import sys
import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier



warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import make_features

TRAIN_PATH = ROOT / "input" / "train.csv"
TEST_PATH = ROOT / "input" / "test.csv"
SAMPLE_PATH = ROOT / "input" / "sample_submission.csv"

OUT = ROOT / "outputs" / "full_experiments"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
FOLDS = 5
THREADS = max(1, min(6, os.cpu_count() or 2))


def auc(y, prediction):
    return float(roc_auc_score(y, prediction))


def safe_feature_names(frame):
    """Sanitize names for LightGBM and XGBoost."""
    names = [
        re.sub(r"[^A-Za-z0-9_]+", "_", str(c))
        for c in frame.columns
    ]

    seen = {}
    unique = []

    for name in names:
        count = seen.get(name, 0)
        unique.append(name if count == 0 else f"{name}_{count}")
        seen[name] = count + 1

    frame.columns = unique
    return frame


def prepare_numeric_features(X, T, cat_cols):
    combined = pd.concat([X, T], ignore_index=True)

    combined = pd.get_dummies(
        combined,
        columns=cat_cols,
        dummy_na=False,
        dtype=np.uint8,
    )

    combined = safe_feature_names(combined)

    X_numeric = combined.iloc[:len(X)].copy()
    T_numeric = combined.iloc[len(X):].copy()

    return X_numeric, T_numeric


def make_signature(X, T, model_name):
    """Prevent stale prediction caches being reused after feature changes."""
    description = {
        "model": model_name,
        "seed": SEED,
        "folds": FOLDS,
        "train_rows": len(X),
        "test_rows": len(T),
        "feature_names": list(X.columns),
        "feature_dtypes": [str(x) for x in X.dtypes],
    }

    raw = json.dumps(
        description,
        sort_keys=True,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()



# Set True only if you want to be asked about reusing older
# prediction files that do not have cache metadata.
ALLOW_LEGACY_CACHE_PROMPT = True


def load_cache(name, signature, n_train, n_test):
    oof_path = OUT / f"oof_{name}.npy"
    test_path = OUT / f"test_{name}.npy"
    meta_path = OUT / f"cache_{name}.json"

    # No prediction files: experiment has not completed.
    if not oof_path.exists() or not test_path.exists():
        return None

    try:
        oof = np.load(oof_path)
        test = np.load(test_path)

        # Check dimensions and valid probability values.
        if oof.shape != (n_train,) or test.shape != (n_test,):
            print(f"{name}: saved predictions have wrong dimensions.")
            return None

        if not np.isfinite(oof).all() or not np.isfinite(test).all():
            print(f"{name}: saved predictions contain invalid values.")
            return None

        if ((oof < 0) | (oof > 1)).any():
            print(f"{name}: invalid OOF probability values.")
            return None

        if ((test < 0) | (test > 1)).any():
            print(f"{name}: invalid test probability values.")
            return None

        # If metadata exists, require an exact signature match.
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())

            if metadata.get("signature") != signature:
                print(
                    f"{name}: experiment configuration/features changed. "
                    "Retraining this model."
                )
                return None

            print(f"{name}: completed experiment found. Skipping training.")
            return oof, test

        # Older cache files have no signature metadata.
        if ALLOW_LEGACY_CACHE_PROMPT:
            print(f"\n{name}: older saved predictions found.")
            print("No metadata exists to verify their feature version.")
            answer = input(
                "Reuse these existing predictions? "
                "Enter Y to reuse or N to retrain: "
            ).strip().lower()

            if answer == "y":
                print(f"{name}: reusing older predictions.")
                save_cache(name, signature, oof, test)
                return oof, test

        print(f"{name}: legacy predictions not reused; retraining.")
        return None

    except Exception as exc:
        print(f"{name}: cache loading failed: {exc}")
        return None


def save_cache(name, signature, oof, test):
    oof_path = OUT / f"oof_{name}.npy"
    test_path = OUT / f"test_{name}.npy"
    meta_path = OUT / f"cache_{name}.json"

    # Save arrays first.
    np.save(oof_path, oof)
    np.save(test_path, test)

    # Write metadata only after both arrays are saved.
    metadata = {
        "model": name,
        "signature": signature,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "oof_rows": int(len(oof)),
        "test_rows": int(len(test)),
        "completed": True,
    }

    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"{name}: predictions and completion metadata saved.")



def fit_predict_model(
    name,
    kind,
    X,
    T,
    Xn,
    Tn,
    y,
    cat_cols,
    splits,
):
    signature = make_signature(X, T, name)

    cached = load_cache(
        name,
        signature,
        len(X),
        len(T),
    )

    if cached is not None:
        return cached[0], cached[1], []

    oof = np.zeros(len(X), dtype=np.float32)
    test_predictions = np.zeros(len(T), dtype=np.float64)

    scores = []
    best_iterations = []

    cat_indices = [
        X.columns.get_loc(c)
        for c in cat_cols
        if c in X.columns
    ]

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        started = time.time()

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        Xn_train = Xn.iloc[train_idx]
        Xn_valid = Xn.iloc[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        if kind.startswith("catboost"):
            depth = 7 if kind == "catboost_d7" else 6

            model = CatBoostClassifier(
                iterations=2000,
                depth=depth,
                learning_rate=0.035,
                loss_function="Logloss",
                eval_metric="AUC",
                l2_leaf_reg=5,
                random_seed=SEED + fold,
                thread_count=THREADS,
                verbose=False,
                allow_writing_files=False,
            )

            model.fit(
                X_train,
                y_train,
                cat_features=cat_indices,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=150,
                verbose=False,
            )

            valid_prediction = model.predict_proba(X_valid)[:, 1]
            test_prediction = model.predict_proba(T)[:, 1]

            best_iteration = int(
                model.best_iteration_ or 2000
            )

        elif kind == "lightgbm":
            model = LGBMClassifier(
                n_estimators=3000,
                learning_rate=0.025,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=3.0,
                reg_alpha=0.1,
                objective="binary",
                random_state=SEED + fold,
                n_jobs=THREADS,
                verbosity=-1,
            )

            model.fit(
                Xn_train,
                y_train,
                eval_set=[(Xn_valid, y_valid)],
                callbacks=[
                    early_stopping(150, verbose=False)
                ],
            )

            valid_prediction = model.predict_proba(Xn_valid)[:, 1]
            test_prediction = model.predict_proba(Tn)[:, 1]

            best_iteration = int(
                model.best_iteration_ or 3000
            )

        elif kind == "xgboost":
            positives = int((y_train == 1).sum())
            negatives = len(y_train) - positives

            model = XGBClassifier(
                n_estimators=3000,
                max_depth=5,
                learning_rate=0.025,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5,
                reg_alpha=0.1,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_jobs=THREADS,
                random_state=SEED + fold,
                scale_pos_weight=negatives / max(positives, 1),
                early_stopping_rounds=150,
            )

            model.fit(
                Xn_train,
                y_train,
                eval_set=[(Xn_valid, y_valid)],
                verbose=False,
            )

            valid_prediction = model.predict_proba(Xn_valid)[:, 1]
            test_prediction = model.predict_proba(Tn)[:, 1]

            best_iteration = int(
                model.best_iteration or 3000
            )

        else:
            raise ValueError(f"Unknown model kind: {kind}")

        oof[valid_idx] = valid_prediction
        test_predictions += test_prediction / FOLDS

        fold_auc = auc(y_valid, valid_prediction)
        scores.append(fold_auc)
        best_iterations.append(best_iteration)

        print(
            f"{name} | fold {fold}/{FOLDS} "
            f"| AUC={fold_auc:.6f} "
            f"| best_iter={best_iteration} "
            f"| seconds={time.time() - started:.1f}",
            flush=True,
        )

        del model
        gc.collect()

    save_cache(name, signature, oof, test_predictions)

    print(
        f"{name} OOF AUC={auc(y, oof):.6f} "
        f"| fold mean={np.mean(scores):.6f}",
        flush=True,
    )

    return oof, test_predictions, best_iterations


def search_blend_weights(y, oofs, model_names):
    """
    Exploratory OOF blend search.

    The score is not an unbiased final estimate because the weights
    are selected using these same OOF predictions.
    """
    best_score = -1
    best_weights = None

    # Coarse simplex grid, weights in increments of 10%.
    for a in range(11):
        for b in range(11 - a):
            for c in range(11 - a - b):
                d = 10 - a - b - c

                weights = np.array(
                    [a, b, c, d],
                    dtype=float,
                ) / 10.0

                prediction = sum(
                    weights[i] * oofs[model_names[i]]
                    for i in range(4)
                )

                score = auc(y, prediction)

                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()

    return best_score, best_weights


def main():
    started = time.time()

    print("Loading Home Credit data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    if not sample["SK_ID_CURR"].equals(test["SK_ID_CURR"]):
        raise ValueError(
            "sample_submission IDs/order do not match test.csv"
        )

    y = train["TARGET"].astype(int).to_numpy()

    # Full model feature set.
    X, T, cat_cols = make_features(
        train,
        test,
        include_employment_sentinel=True,
    )

    # Ablation feature set: remove only the sentinel indicator.
    X_no_sentinel, T_no_sentinel, cat_no_sentinel = make_features(
        train,
        test,
        include_employment_sentinel=False,
    )

    print(f"Train rows: {len(X):,}")
    print(f"Test rows: {len(T):,}")
    print(f"CatBoost features: {X.shape[1]:,}")

    Xn, Tn = prepare_numeric_features(X, T, cat_cols)

    # The ablation only needs CatBoost, so it uses the categorical input.
    skf = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    splits = list(skf.split(X, y))

    configurations = [
        ("catboost_d6", "catboost_d6", X, T, cat_cols),
        ("catboost_d7", "catboost_d7", X, T, cat_cols),
        (
            "catboost_d6_no_sentinel",
            "catboost_d6",
            X_no_sentinel,
            T_no_sentinel,
            cat_no_sentinel,
        ),
        ("lightgbm", "lightgbm", X, T, cat_cols),
        ("xgboost", "xgboost", X, T, cat_cols),
    ]

    oofs = {}
    tests = {}
    model_results = {}

    for name, kind, model_X, model_T, model_cats in configurations:
        print(f"\n========== {name} ==========", flush=True)

        if kind == "lightgbm" or kind == "xgboost":
            model_Xn, model_Tn = Xn, Tn
        elif name == "catboost_d6_no_sentinel":
            model_Xn, model_Tn = Xn, Tn
        else:
            model_Xn, model_Tn = Xn, Tn

        oof, test_prediction, best_iterations = fit_predict_model(
            name=name,
            kind=kind,
            X=model_X,
            T=model_T,
            Xn=model_Xn,
            Tn=model_Tn,
            y=y,
            cat_cols=model_cats,
            splits=splits,
        )

        oofs[name] = oof
        tests[name] = test_prediction

        model_results[name] = {
            "oof_auc": auc(y, oof),
            "best_iterations": best_iterations,
        }

    # ---------------------------------------------------------
    # Sentinel ablation comparison
    # ---------------------------------------------------------
    full_score = model_results["catboost_d6"]["oof_auc"]
    ablation_score = model_results[
        "catboost_d6_no_sentinel"
    ]["oof_auc"]

    sentinel_report = {
        "catboost_d6_with_sentinel_auc": full_score,
        "catboost_d6_without_sentinel_auc": ablation_score,
        "difference_with_minus_without": full_score - ablation_score,
        "interpretation": (
            "Positive difference means the sentinel feature improved "
            "this OOF result; negative means it reduced it. "
            "This is a single-seed comparison, not proof of causality."
        ),
    }

    (OUT / "sentinel_ablation.json").write_text(
        json.dumps(sentinel_report, indent=2)
    )

    print("\nEMPLOYMENT SENTINEL ABLATION")
    print(json.dumps(sentinel_report, indent=2))

    # ---------------------------------------------------------
    # Blend only the four main models, not the ablation model.
    # ---------------------------------------------------------
    blend_names = [
        "catboost_d6",
        "catboost_d7",
        "lightgbm",
        "xgboost",
    ]

    blend_auc, blend_weights = search_blend_weights(
        y,
        oofs,
        blend_names,
    )

    blend_oof = sum(
        blend_weights[i] * oofs[name]
        for i, name in enumerate(blend_names)
    )

    blend_test = sum(
        blend_weights[i] * tests[name]
        for i, name in enumerate(blend_names)
    )

    if not np.isfinite(blend_test).all():
        raise ValueError("Blend contains non-finite probabilities.")

    if ((blend_test < 0) | (blend_test > 1)).any():
        raise ValueError("Blend probabilities are outside [0, 1].")

    submission = sample.copy()
    submission["TARGET"] = blend_test

    submission_path = ROOT / "outputs" / "submission_v2_1.csv"
    submission_path.parent.mkdir(exist_ok=True)
    submission.to_csv(submission_path, index=False)

    # Save OOF predictions and test predictions.
    oof_frame = pd.DataFrame({
        "TARGET": y,
        **{f"oof_{name}": values for name, values in oofs.items()},
        "oof_blend": blend_oof,
    })

    oof_frame.to_csv(
        OUT / "oof_predictions.csv",
        index=False,
    )

    test_frame = pd.DataFrame({
        "SK_ID_CURR": sample["SK_ID_CURR"],
        **{f"test_{name}": values for name, values in tests.items()},
        "test_blend": blend_test,
    })

    test_frame.to_csv(
        OUT / "test_predictions.csv",
        index=False,
    )

    report = {
        "seed": SEED,
        "folds": FOLDS,
        "train_rows": len(X),
        "test_rows": len(T),
        "features": X.shape[1],
        "one_hot_features": Xn.shape[1],
        "models": model_results,
        "sentinel_ablation": sentinel_report,
        "blend": {
            "models": blend_names,
            "weights": {
                name: float(weight)
                for name, weight in zip(
                    blend_names,
                    blend_weights,
                )
            },
            "oof_selection_auc": float(blend_auc),
            "warning": (
                "Blend weights were optimized on these OOF predictions. "
                "The resulting blend score is a model-selection score "
                "and may be optimistic."
            ),
        },
        "runtime_seconds": round(time.time() - started, 1),
        "submission": str(submission_path.relative_to(ROOT)),
    }

    (OUT / "experiment_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nFINAL EXPERIMENT REPORT")
    print(json.dumps(report, indent=2))
    print(f"\nSubmission saved to: {submission_path}")


if __name__ == "__main__":
    main()
