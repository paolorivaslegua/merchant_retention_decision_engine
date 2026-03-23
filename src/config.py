from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"
MPL_CONFIG_DIR = CACHE_DIR / "matplotlib"

RAW_DATA_PATH = RAW_DIR / "merchants.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "merchant_features.csv"
MERCHANT_OUTPUT_PATH = TABLES_DIR / "merchant_retention_actions.csv"
METRICS_OUTPUT_PATH = TABLES_DIR / "model_metrics.csv"
ACTION_SUMMARY_PATH = TABLES_DIR / "action_summary.csv"
TOP_PRIORITIES_PATH = TABLES_DIR / "top_priority_merchants.csv"
COEFFICIENTS_OUTPUT_PATH = TABLES_DIR / "model_coefficients.csv"
MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"

DEFAULT_SYNTHETIC_ROWS = 3000
RANDOM_SEED = 42
TEST_SIZE = 0.25

RETENTION_HORIZON_MONTHS = 6
GROSS_MARGIN_RATE = 0.012
INCENTIVE_COST = 250.0

ACTION_UPLIFT = {
    "priority_outreach": 0.35,
    "offer_incentive": 0.28,
    "product_education": 0.18,
    "monitor_only": 0.05,
}

ACTION_THRESHOLDS = {
    "priority_risk": 0.58,
    "offer_risk": 0.42,
    "education_risk": 0.28,
    "priority_gpv": 30000.0,
    "severe_chargeback_rate": 0.016,
    "high_inactivity_days": 40,
    "low_product_adoption": 2,
}

SEGMENT_GPV = {
    "micro": (4000, 2500),
    "smb": (18000, 9000),
    "mid_market": (65000, 22000),
    "enterprise": (160000, 45000),
}

SEGMENT_WEIGHTS = {
    "micro": 0.35,
    "smb": 0.40,
    "mid_market": 0.20,
    "enterprise": 0.05,
}


def ensure_directories() -> None:
    """Create project directories required by the pipeline."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        MPL_CONFIG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
