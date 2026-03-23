from __future__ import annotations

import pandas as pd

from . import config


def recommend_action(row: pd.Series) -> str:
    """Translate churn risk into a retention action."""
    thresholds = config.ACTION_THRESHOLDS
    churn_probability = row["churn_probability"]

    if churn_probability >= thresholds["priority_risk"] and (
        row["monthly_gpv"] >= thresholds["priority_gpv"]
        or row["chargeback_rate"] >= thresholds["severe_chargeback_rate"]
        or row["inactivity_days"] >= thresholds["high_inactivity_days"]
    ):
        return "priority_outreach"

    if churn_probability >= thresholds["offer_risk"] and row["monthly_gpv"] >= 12000:
        return "offer_incentive"

    if (
        churn_probability >= thresholds["education_risk"]
        and row["product_adoption_count"] <= thresholds["low_product_adoption"]
    ):
        return "product_education"

    return "monitor_only"


def compute_expected_retention_value(row: pd.Series) -> float:
    """Estimate expected retained gross margin from the recommended action."""
    action = row["recommended_action"]
    gross_margin_value = (
        row["monthly_gpv"] * config.GROSS_MARGIN_RATE * config.RETENTION_HORIZON_MONTHS
    )
    expected_value = gross_margin_value * row["churn_probability"] * config.ACTION_UPLIFT[action]

    if action == "offer_incentive":
        expected_value -= config.INCENTIVE_COST

    return round(max(expected_value, 0.0), 2)


def apply_decision_engine(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Assign actions, estimate value, and prioritize the merchant queue."""
    df = scored_df.copy()
    df["recommended_action"] = df.apply(recommend_action, axis=1)
    df["expected_retention_value"] = df.apply(compute_expected_retention_value, axis=1)

    df = df.sort_values(
        by=["expected_retention_value", "churn_probability", "monthly_gpv"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df["priority_rank"] = df.index + 1
    return df
