from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "heart_disease_processed.csv"


def test_processed_data_exists():
    assert DATA_PATH.exists()


def test_processed_data_can_be_loaded():
    df = pd.read_csv(DATA_PATH)
    assert not df.empty


def test_processed_data_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0


def test_target_column_exists():
    df = pd.read_csv(DATA_PATH)
    assert "target" in df.columns


def test_processed_data_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (303, 21)