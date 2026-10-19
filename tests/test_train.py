from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "heart_disease_processed.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_model.joblib"


def test_training_data_exists():
    assert DATA_PATH.exists()


def test_training_data_has_target():
    df = pd.read_csv(DATA_PATH)
    assert "target" in df.columns


def test_training_data_has_expected_rows():
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 303


def test_selected_model_exists():
    assert MODEL_PATH.exists()


def test_model_comparison_file_exists():
    comparison_path = PROJECT_ROOT / "model_comparison.csv"
    assert comparison_path.exists()


def test_model_comparison_contains_all_models():
    comparison_path = PROJECT_ROOT / "model_comparison.csv"
    df = pd.read_csv(comparison_path)

    expected_models = {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    }

    assert expected_models.issubset(set(df["Model"]))