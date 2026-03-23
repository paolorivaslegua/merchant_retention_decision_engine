from __future__ import annotations

import os

from . import config
from .data_generation import load_or_generate_data
from .decision_engine import apply_decision_engine
from .features import FEATURE_COLUMNS, build_feature_dataset
from .model import train_and_score

os.environ.setdefault("MPLCONFIGDIR", str(config.MPL_CONFIG_DIR))

from .evaluation import save_figures, save_metrics_and_tables


def main() -> None:
    """Run the full merchant retention workflow."""
    config.ensure_directories()

    raw_df = load_or_generate_data()
    feature_df = build_feature_dataset(raw_df)
    feature_df.to_csv(config.PROCESSED_DATA_PATH, index=False)

    model_artifacts = train_and_score(feature_df, FEATURE_COLUMNS)
    scored_df = apply_decision_engine(model_artifacts.scored_df)
    scored_df.to_csv(config.MERCHANT_OUTPUT_PATH, index=False)

    tables = save_metrics_and_tables(
        scored_df=scored_df,
        metrics_df=model_artifacts.metrics_df,
        coefficients_df=model_artifacts.coefficients_df,
    )
    save_figures(
        scored_df=scored_df,
        action_summary=tables["action_summary"],
        coefficients_df=model_artifacts.coefficients_df,
    )

    metrics_row = model_artifacts.metrics_df.iloc[0]
    total_expected_value = scored_df["expected_retention_value"].sum()
    action_counts = scored_df["recommended_action"].value_counts().to_dict()

    print("Merchant Retention Decision Engine Summary")
    print(f"Merchants scored: {len(scored_df)}")
    print(f"Observed churn rate: {raw_df['churned'].mean():.2%}")
    print(f"Model ROC-AUC: {metrics_row['roc_auc']:.3f}")
    print(f"Action counts: {action_counts}")
    print(f"Total expected retention value: ${total_expected_value:,.2f}")
    print(f"Top merchant by priority: {scored_df.iloc[0]['merchant_id']}")


if __name__ == "__main__":
    main()
