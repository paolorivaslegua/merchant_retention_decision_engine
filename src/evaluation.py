from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from . import config


def save_metrics_and_tables(
    scored_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Persist metrics and summary tables for downstream use."""
    action_summary = (
        scored_df.groupby("recommended_action", as_index=False)
        .agg(
            merchant_count=("merchant_id", "count"),
            observed_churn_rate=("churned", "mean"),
            average_churn_probability=("churn_probability", "mean"),
            total_expected_retention_value=("expected_retention_value", "sum"),
        )
        .sort_values("total_expected_retention_value", ascending=False)
    )

    top_priorities = scored_df[
        [
            "merchant_id",
            "segment",
            "monthly_gpv",
            "churn_probability",
            "recommended_action",
            "expected_retention_value",
            "priority_rank",
        ]
    ].head(25)

    metrics_df.to_csv(config.METRICS_OUTPUT_PATH, index=False)
    action_summary.to_csv(config.ACTION_SUMMARY_PATH, index=False)
    top_priorities.to_csv(config.TOP_PRIORITIES_PATH, index=False)
    coefficients_df.to_csv(config.COEFFICIENTS_OUTPUT_PATH, index=False)

    return {
        "action_summary": action_summary,
        "top_priorities": top_priorities,
    }


def _save_probability_distribution(scored_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(scored_df["churn_probability"], bins=25, color="#1f77b4", edgecolor="white")
    plt.title("Churn Probability Distribution")
    plt.xlabel("Churn Probability")
    plt.ylabel("Merchant Count")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "churn_probability_distribution.png", dpi=150)
    plt.close()


def _save_action_distribution(action_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(
        action_summary["recommended_action"],
        action_summary["merchant_count"],
        color="#2ca02c",
    )
    plt.title("Recommended Action Distribution")
    plt.xlabel("Recommended Action")
    plt.ylabel("Merchant Count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "action_distribution.png", dpi=150)
    plt.close()


def _save_value_by_action(action_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(
        action_summary["recommended_action"],
        action_summary["total_expected_retention_value"],
        color="#ff7f0e",
    )
    plt.title("Total Expected Retention Value by Action")
    plt.xlabel("Recommended Action")
    plt.ylabel("Expected Retention Value")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "expected_retention_value_by_action.png", dpi=150)
    plt.close()


def _save_feature_coefficients(coefficients_df: pd.DataFrame) -> None:
    top_coefficients = coefficients_df.head(10).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top_coefficients["feature"], top_coefficients["coefficient"], color="#d62728")
    plt.title("Top Logistic Regression Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "top_feature_coefficients.png", dpi=150)
    plt.close()


def _save_risk_value_scatter(scored_df: pd.DataFrame) -> None:
    action_colors = {
        "priority_outreach": "#d62728",
        "offer_incentive": "#ff7f0e",
        "product_education": "#1f77b4",
        "monitor_only": "#7f7f7f",
    }

    plt.figure(figsize=(8, 5))
    for action, action_df in scored_df.groupby("recommended_action"):
        plt.scatter(
            action_df["monthly_gpv"],
            action_df["churn_probability"],
            label=action,
            alpha=0.55,
            s=20,
            color=action_colors[action],
        )

    plt.title("Churn Probability vs Monthly GPV")
    plt.xlabel("Monthly GPV")
    plt.ylabel("Churn Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "risk_vs_gpv_scatter.png", dpi=150)
    plt.close()


def save_figures(
    scored_df: pd.DataFrame,
    action_summary: pd.DataFrame,
    coefficients_df: pd.DataFrame,
) -> None:
    """Create portfolio-friendly figures."""
    _save_probability_distribution(scored_df)
    _save_action_distribution(action_summary)
    _save_value_by_action(action_summary)
    _save_feature_coefficients(coefficients_df)
    _save_risk_value_scatter(scored_df)
