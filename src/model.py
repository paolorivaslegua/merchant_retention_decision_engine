from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class ModelArtifacts:
    model: Pipeline
    scored_df: pd.DataFrame
    metrics_df: pd.DataFrame
    coefficients_df: pd.DataFrame


def train_and_score(feature_df: pd.DataFrame, feature_columns: list[str]) -> ModelArtifacts:
    """Train the churn model, persist it, and score the full merchant base."""
    X = feature_df[feature_columns].copy()
    y = feature_df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)),
        ]
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    all_probs = model.predict_proba(X)[:, 1]

    metrics_df = pd.DataFrame(
        [
            {
                "roc_auc": round(roc_auc_score(y_test, test_probs), 4),
                "precision": round(precision_score(y_test, test_preds, zero_division=0), 4),
                "recall": round(recall_score(y_test, test_preds, zero_division=0), 4),
                "f1": round(f1_score(y_test, test_preds, zero_division=0), 4),
                "average_predicted_churn": round(all_probs.mean(), 4),
            }
        ]
    )

    coefficients = model.named_steps["classifier"].coef_[0]
    coefficients_df = (
        pd.DataFrame({"feature": feature_columns, "coefficient": coefficients})
        .sort_values("coefficient", key=lambda series: series.abs(), ascending=False)
        .reset_index(drop=True)
    )

    scored_df = feature_df.copy()
    scored_df["churn_probability"] = all_probs

    joblib.dump(model, config.MODEL_PATH)
    return ModelArtifacts(
        model=model,
        scored_df=scored_df,
        metrics_df=metrics_df,
        coefficients_df=coefficients_df,
    )
