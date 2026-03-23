from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_gpv",
    "gpv_trend_pct",
    "chargeback_rate",
    "support_tickets_90d",
    "product_adoption_count",
    "inactivity_days",
    "segment_micro",
    "segment_smb",
    "segment_mid_market",
    "segment_enterprise",
    "log_monthly_gpv",
    "tickets_per_tenure_month",
]


def build_feature_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create a business-readable modeling frame."""
    df = raw_df.copy()
    df["log_monthly_gpv"] = np.log1p(df["monthly_gpv"])
    df["tickets_per_tenure_month"] = (
        df["support_tickets_90d"] / df["tenure_months"].clip(lower=1)
    ).round(4)

    segment_dummies = pd.get_dummies(df["segment"], prefix="segment")
    for column in (
        "segment_micro",
        "segment_smb",
        "segment_mid_market",
        "segment_enterprise",
    ):
        if column not in segment_dummies.columns:
            segment_dummies[column] = 0

    df = pd.concat([df, segment_dummies], axis=1)
    return df


def get_model_inputs(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split the feature frame into identifiers, features, and target."""
    X = feature_df[FEATURE_COLUMNS].copy()
    y = feature_df["churned"].astype(int)
    merchant_ids = feature_df["merchant_id"]
    return X, y, merchant_ids
