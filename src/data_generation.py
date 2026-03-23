from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _bounded_normal(rng: np.random.Generator, mean: float, std: float, minimum: float) -> float:
    """Sample a positive-ish number without creating unrealistic negatives."""
    return max(rng.normal(mean, std), minimum)


def generate_synthetic_data(
    num_rows: int = config.DEFAULT_SYNTHETIC_ROWS,
    random_seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a merchant portfolio with correlated risk signals."""
    rng = np.random.default_rng(random_seed)
    segments = list(config.SEGMENT_WEIGHTS.keys())
    segment_probs = list(config.SEGMENT_WEIGHTS.values())

    records: list[dict[str, float | int | str]] = []
    for merchant_num in range(1, num_rows + 1):
        segment = rng.choice(segments, p=segment_probs)
        gpv_mean, gpv_std = config.SEGMENT_GPV[segment]

        tenure_months = int(np.clip(rng.gamma(shape=3.0, scale=8.0), 1, 120))
        monthly_gpv = _bounded_normal(rng, gpv_mean, gpv_std, 500.0)
        inactivity_days = int(np.clip(rng.normal(12, 10), 0, 120))

        trend_mean = {
            "micro": -0.015,
            "smb": 0.002,
            "mid_market": 0.006,
            "enterprise": 0.01,
        }[segment]
        gpv_trend_pct = float(np.clip(rng.normal(trend_mean, 0.08), -0.35, 0.30))

        base_chargeback = {
            "micro": 0.011,
            "smb": 0.008,
            "mid_market": 0.006,
            "enterprise": 0.005,
        }[segment]
        chargeback_rate = float(np.clip(rng.normal(base_chargeback, 0.004), 0.0005, 0.05))

        product_adoption_count = int(
            np.clip(
                rng.poisson(lam=1.2 + tenure_months / 24 + (segment in {"mid_market", "enterprise"})),
                1,
                8,
            )
        )

        support_lambda = (
            1.5
            + (0.8 if gpv_trend_pct < -0.05 else 0.0)
            + (1.0 if chargeback_rate > 0.015 else 0.0)
            + max(0, 3 - product_adoption_count) * 0.35
        )
        support_tickets_90d = int(np.clip(rng.poisson(support_lambda), 0, 20))

        churn_logit = (
            -1.12
            - 0.016 * tenure_months
            - 0.26 * product_adoption_count
            + 11.5 * chargeback_rate
            + 0.14 * support_tickets_90d
            + 0.034 * inactivity_days
            - 3.9 * gpv_trend_pct
            - 0.000004 * monthly_gpv
            + rng.normal(0, 0.65)
        )
        churn_probability = 1 / (1 + np.exp(-churn_logit))
        churned = int(rng.binomial(1, np.clip(churn_probability, 0.01, 0.98)))

        records.append(
            {
                "merchant_id": f"M{merchant_num:05d}",
                "segment": segment,
                "tenure_months": tenure_months,
                "monthly_gpv": round(monthly_gpv, 2),
                "gpv_trend_pct": round(gpv_trend_pct, 4),
                "chargeback_rate": round(chargeback_rate, 4),
                "support_tickets_90d": support_tickets_90d,
                "product_adoption_count": product_adoption_count,
                "inactivity_days": inactivity_days,
                "churned": churned,
            }
        )

    return pd.DataFrame(records)


def load_or_generate_data() -> pd.DataFrame:
    """Read raw data if present; otherwise generate and save it."""
    if config.RAW_DATA_PATH.exists():
        return pd.read_csv(config.RAW_DATA_PATH)

    config.ensure_directories()
    data = generate_synthetic_data()
    data.to_csv(config.RAW_DATA_PATH, index=False)
    return data
