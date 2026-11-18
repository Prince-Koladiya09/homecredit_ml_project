"""Deterministic, leakage-safe feature construction for the supplied Home Credit table."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class HomeCreditFeatureBuilder(BaseEstimator, TransformerMixin):
    """Add interpretable ratios and date-derived features without using TARGET."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps = 1e-6
        if "DAYS_BIRTH" in X:
            X["AGE_YEARS"] = (-X["DAYS_BIRTH"] / 365.25).clip(lower=0)
        if "DAYS_EMPLOYED" in X:
            # The large positive value is a known sentinel in this dataset.
            raw = pd.to_numeric(X["DAYS_EMPLOYED"], errors="coerce")
            X["EMPLOYMENT_DAYS_SENTINEL"] = (raw > 0).astype("int8")
            X["EMPLOYED_YEARS"] = (-raw.where(raw <= 0) / 365.25).clip(lower=0)
        pairs = [
            ("AMT_CREDIT", "AMT_INCOME_TOTAL", "CREDIT_TO_INCOME"),
            ("AMT_ANNUITY", "AMT_INCOME_TOTAL", "ANNUITY_TO_INCOME"),
            ("AMT_CREDIT", "AMT_ANNUITY", "CREDIT_TO_ANNUITY"),
            ("AMT_GOODS_PRICE", "AMT_CREDIT", "GOODS_TO_CREDIT"),
        ]
        for numerator, denominator, name in pairs:
            if numerator in X and denominator in X:
                den = pd.to_numeric(X[denominator], errors="coerce").replace(0, np.nan)
                X[name] = pd.to_numeric(X[numerator], errors="coerce") / (den + eps)
        ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in X]
        if ext:
            X["EXT_SOURCE_MEAN"] = X[ext].mean(axis=1)
            X["EXT_SOURCE_STD"] = X[ext].std(axis=1)
            X["EXT_SOURCE_MISSING_COUNT"] = X[ext].isna().sum(axis=1)
        # Identifier is for joining predictions, not a customer-risk feature.
        X = X.drop(columns=["SK_ID_CURR"], errors="ignore")
        return X
