from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "heart_disease_processed.csv"


def load_data():
    """Load the processed heart disease dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


def split_features_target(df):
    """Separate input features (X) from target (y)."""

    # Target column in the processed dataset
    target_column = "target"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()

    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    X, y = split_features_target(df)

    print("\nFeature shape:", X.shape)
    print("Target shape:", y.shape)

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTesting Features:")
print(X_test)

print("\nTesting Actual Target:")
print(y_test)