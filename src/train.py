from pathlib import Path


import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier



# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "heart_disease_processed.csv"


def load_data():
    """Load the preprocessed dataset."""
    return pd.read_csv(DATA_PATH)


def split_features_target(df):
    """Separate features and target."""

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def create_train_test_split(X, y):
    """Create reproducible stratified train/test split."""

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with feature scaling."""

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    return model

def train_decision_tree(X_train, y_train):
    """Train a Decision Tree model."""

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=4
    )

    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest model."""

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate a trained model and return key classification metrics."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{model_name} Evaluation")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

def cross_validate_model(model_name, model, X_train, y_train):
    """Perform 5-fold stratified cross-validation."""
    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    print(f"\n{model_name} 5-Fold Cross-Validation")
    print("-" * 40)
    print(f"Accuracy : {cv_results['test_accuracy'].mean():.4f}")
    print(f"Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"Recall   : {cv_results['test_recall'].mean():.4f}")
    print(f"F1 Score : {cv_results['test_f1'].mean():.4f}")
    print(f"ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f}")

    return {
        "Model": model_name,
        "Accuracy": cv_results["test_accuracy"].mean(),
        "Precision": cv_results["test_precision"].mean(),
        "Recall": cv_results["test_recall"].mean(),
        "F1 Score": cv_results["test_f1"].mean(),
        "ROC-AUC": cv_results["test_roc_auc"].mean()
    }

if __name__ == "__main__":

    mlflow.set_experiment("heart-disease-classification")

    print("Loading dataset...")

    df = load_data()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining Logistic Regression...")

    logistic_model = train_logistic_regression(X_train, y_train)

    print("Logistic Regression training completed.")

    #decision tree training    
    print("\nTraining Decision Tree...")

    decision_tree = train_decision_tree(X_train, y_train)

    print("Decision Tree training completed.")


    print("\nTraining Random Forest...")
    random_forest = train_random_forest(X_train, y_train)

    print("Random Forest training completed.")


    results = []

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            X_test,
            y_test
        )
    )

    results.append(
        evaluate_model(
            "Decision Tree",
            decision_tree,
            X_test,
            y_test
        )
    )

    results.append(
        evaluate_model(
            "Random Forest",
            random_forest,
            X_test,
            y_test
        )
    )

    results_df = pd.DataFrame(results)

    print("\nModel Comparison")
    print("================")
    print(results_df.round(4).to_string(index=False))

    # Random Forest Confusion Matrix
    y_pred_rf = random_forest.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_rf)

    print("\nRandom Forest Confusion Matrix")
    print("==============================")
    print(cm)

        # 5-Fold Stratified Cross-Validation
    print("\n\nStarting 5-Fold Stratified Cross-Validation...")

    cv_results = []

    cv_results.append(
        cross_validate_model(
            "Logistic Regression",
            logistic_model,
            X_train,
            y_train
        )
    )

    cv_results.append(
        cross_validate_model(
            "Decision Tree",
            decision_tree,
            X_train,
            y_train
        )
    )

    cv_results.append(
        cross_validate_model(
            "Random Forest",
            random_forest,
            X_train,
            y_train
        )
    )

    cv_results_df = pd.DataFrame(cv_results)

    print("\nCross-Validation Model Comparison")
    print("=================================")
    print(cv_results_df.round(4).to_string(index=False))

        # Final evaluation of selected model
    print("\n\nFinal Evaluation - Logistic Regression")
    print("======================================")

    y_pred_final = logistic_model.predict(X_test)
    y_prob_final = logistic_model.predict_proba(X_test)[:, 1]

    final_accuracy = accuracy_score(y_test, y_pred_final)
    final_precision = precision_score(y_test, y_pred_final)
    final_recall = recall_score(y_test, y_pred_final)
    final_f1 = f1_score(y_test, y_pred_final)
    final_roc_auc = roc_auc_score(y_test, y_prob_final)

    print(f"Accuracy : {final_accuracy:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall   : {final_recall:.4f}")
    print(f"F1 Score : {final_f1:.4f}")
    print(f"ROC-AUC  : {final_roc_auc:.4f}")

    final_cm = confusion_matrix(y_test, y_pred_final)

    print("\nLogistic Regression Confusion Matrix")
    print("====================================")
    print(final_cm)
        # Save the final selected model
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "logistic_regression_model.joblib"

    joblib.dump(logistic_model, model_path)

    print("\nFinal model saved successfully.")
    print(f"Model path: {model_path}")