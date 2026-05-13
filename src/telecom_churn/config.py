from pathlib import Path

DEFAULT_RANDOM_STATE = 23
DEFAULT_TEST_SIZE = 0.2
# Fraction of the pre-test fit set held out for validation (Optuna / early stopping), stratified.
DEFAULT_VAL_FRACTION_OF_FIT = 0.25
DEFAULT_OPTUNA_TRIALS = 100

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "mobile-churn-data.xlsx"
