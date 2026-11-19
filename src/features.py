
"""Leakage-conscious feature engineering for Home Credit."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _income_band_edges(train_income: pd.Series) -> np.ndarray:
    """Calculate income-band boundaries using training data only."""
    values = pd.to_numeric(train_income, errors="coerce").dropna()

    if values.empty:
        return np.array([-np.inf, np.inf])

    quantiles = values.quantile([0.20, 0.40, 0.60, 0.80]).to_numpy()
    edges = np.unique(np.concatenate(([-np.inf], quantiles, [np.inf])))

    if len(edges) < 2:
        return np.array([-np.inf, np.inf])

    return edges


def make_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    include_employment_sentinel: bool = True,
):
    """
    Build train/test features consistently.

    Returns:
        X: engineered training features
        T: engineered test features
        cat_cols: categorical feature names
    """
    if "TARGET" not in train.columns:
        raise ValueError("Training data must contain TARGET.")

    tr = train.drop(columns=["TARGET"]).copy()
    te = test.copy()

    if "SK_ID_CURR" not in tr or "SK_ID_CURR" not in te:
        raise ValueError("SK_ID_CURR is required in train and test.")

    # Preserve test IDs outside the model features for submission joining.
    n_train = len(tr)
    both = pd.concat([tr, te], ignore_index=True)

    eps = 1e-6

    # ---------------------------------------------------------
    # Age and employment features
    # ---------------------------------------------------------
    if "DAYS_BIRTH" in both:
        both["AGE_YEARS"] = (
            -pd.to_numeric(both["DAYS_BIRTH"], errors="coerce") / 365.25
        )

    if "DAYS_EMPLOYED" in both:
        employed_days = pd.to_numeric(
            both["DAYS_EMPLOYED"], errors="coerce"
        )

        if include_employment_sentinel:
            both["EMPLOYMENT_SENTINEL"] = (
                employed_days > 0
            ).astype("int8")

        both["EMPLOYED_YEARS"] = (
            -employed_days.where(employed_days <= 0) / 365.25
        )

        if "AGE_YEARS" in both:
            both["EMPLOYED_TO_AGE"] = (
                both["EMPLOYED_YEARS"] /
                (both["AGE_YEARS"] + eps)
            )

    for original, engineered in [
        ("DAYS_REGISTRATION", "REGISTRATION_YEARS"),
        ("DAYS_ID_PUBLISH", "ID_PUBLISH_YEARS"),
        ("DAYS_LAST_PHONE_CHANGE", "PHONE_CHANGE_YEARS"),
    ]:
        if original in both:
            both[engineered] = (
                -pd.to_numeric(both[original], errors="coerce") / 365.25
            )

    # ---------------------------------------------------------
    # Financial ratios
    # ---------------------------------------------------------
    ratio_definitions = [
        ("AMT_CREDIT", "AMT_INCOME_TOTAL", "CREDIT_TO_INCOME"),
        ("AMT_ANNUITY", "AMT_INCOME_TOTAL", "ANNUITY_TO_INCOME"),
        ("AMT_CREDIT", "AMT_ANNUITY", "CREDIT_TO_ANNUITY"),
        ("AMT_GOODS_PRICE", "AMT_CREDIT", "GOODS_TO_CREDIT"),
        ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "INCOME_PER_FAMILY"),
        ("AMT_CREDIT", "CNT_FAM_MEMBERS", "CREDIT_PER_FAMILY"),
        ("OWN_CAR_AGE", "DAYS_BIRTH", "CAR_AGE_TO_BIRTH_DAYS"),
    ]

    for numerator, denominator, new_name in ratio_definitions:
        if numerator in both and denominator in both:
            num = pd.to_numeric(both[numerator], errors="coerce")
            den = pd.to_numeric(both[denominator], errors="coerce")
            den = den.replace(0, np.nan)
            both[new_name] = num / (den + eps)

    # ---------------------------------------------------------
    # Income bands: boundaries learned from train only
    # ---------------------------------------------------------
    if "AMT_INCOME_TOTAL" in both:
        income_edges = _income_band_edges(
            train["AMT_INCOME_TOTAL"]
        )

        income = pd.to_numeric(
            both["AMT_INCOME_TOTAL"], errors="coerce"
        )

        both["INCOME_BAND"] = pd.cut(
            income,
            bins=income_edges,
            labels=False,
            include_lowest=True,
        ).astype("float")

        both["INCOME_LOG1P"] = np.log1p(
            income.clip(lower=0)
        )

    # ---------------------------------------------------------
    # External source features
    # ---------------------------------------------------------
    ext_cols = [
        c for c in
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        if c in both.columns
    ]

    if ext_cols:
        both["EXT_SOURCE_MEAN"] = both[ext_cols].mean(axis=1)
        both["EXT_SOURCE_MIN"] = both[ext_cols].min(axis=1)
        both["EXT_SOURCE_MAX"] = both[ext_cols].max(axis=1)
        both["EXT_SOURCE_STD"] = both[ext_cols].std(axis=1)
        both["EXT_SOURCE_MISSING_COUNT"] = (
            both[ext_cols].isna().sum(axis=1)
        )

        for i, first in enumerate(ext_cols):
            for second in ext_cols[i + 1:]:
                both[f"{first}_{second}_PRODUCT"] = (
                    both[first] * both[second]
                )
                both[f"{first}_{second}_DIFF"] = (
                    both[first] - both[second]
                )

    # ---------------------------------------------------------
    # Organization XNA indicator
    # ---------------------------------------------------------
    if "ORGANIZATION_TYPE" in both:
        organization = (
            both["ORGANIZATION_TYPE"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        both["ORGANIZATION_TYPE_XNA"] = (
            organization == "XNA"
        ).astype("int8")

    # ---------------------------------------------------------
    # Missingness aggregates
    # ---------------------------------------------------------
    missing_groups = {
        "AMOUNT": [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "OWN_CAR_AGE",
        ],
        "DAYS": [
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_REGISTRATION",
            "DAYS_ID_PUBLISH",
            "DAYS_LAST_PHONE_CHANGE",
        ],
        "EXTERNAL": ext_cols,
        "BUREAU": [
            c for c in both.columns
            if c.startswith("AMT_REQ_CREDIT_BUREAU_")
        ],
    }

    for group_name, columns in missing_groups.items():
        columns = [c for c in columns if c in both.columns]
        if columns:
            both[f"{group_name}_MISSING_COUNT"] = (
                both[columns].isna().sum(axis=1)
            )

    flag_cols = [
        c for c in both.columns
        if c.startswith("FLAG_")
        and pd.api.types.is_numeric_dtype(both[c])
    ]

    if flag_cols:
        both["FLAG_SUM"] = both[flag_cols].sum(axis=1)

    # ---------------------------------------------------------
    # Age bands and amount transformations
    # ---------------------------------------------------------
    if "AGE_YEARS" in both:
        both["AGE_BAND"] = pd.cut(
            both["AGE_YEARS"],
            bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 70, 100],
            labels=False,
            include_lowest=True,
        )

    for column in [
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ]:
        if column in both:
            values = pd.to_numeric(both[column], errors="coerce")
            both[f"{column}_LOG1P"] = np.log1p(
                values.clip(lower=0)
            )

    # ---------------------------------------------------------
    # Categorical missing values and frequency encoding
    # ---------------------------------------------------------
    cat_cols = both.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    for column in cat_cols:
        values = both[column].astype("string").fillna("__MISSING__")
        both[column] = values.astype(str)

        # Frequency is calculated without TARGET.
        frequencies = both[column].value_counts(normalize=True)
        both[f"{column}_FREQUENCY"] = (
            both[column].map(frequencies).astype("float32")
        )

    # ID is used only to align submission rows.
    both = both.drop(columns=["SK_ID_CURR"], errors="ignore")

    X = both.iloc[:n_train].reset_index(drop=True)
    T = both.iloc[n_train:].reset_index(drop=True)

    return X, T, cat_cols
