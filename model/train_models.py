from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_utils import (  # noqa: E402
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    calculate_metrics,
    get_models,
    load_dry_bean_dataset,
    normalize_columns,
    save_model,
)


MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


def main():
    print("=" * 72)
    print("ML Assignment 2 - Dry Bean Classification")
    print("=" * 72)

    print("\n[1/7] Loading dataset...")
    df = load_dry_bean_dataset(PROJECT_ROOT)
    df = normalize_columns(df)

    # Drop only rows that are incomplete in required fields.
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    before = len(df)
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    dropped = before - len(df)

    print(f"Dataset shape          : {df.shape}")
    print(f"Input features         : {len(FEATURE_COLUMNS)}")
    print(f"Target column          : {TARGET_COLUMN}")
    print(f"Number of classes      : {df[TARGET_COLUMN].nunique()}")
    print(f"Missing rows removed   : {dropped}")
    print("\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    print("\n[2/7] Preparing X and y...")
    X = df[FEATURE_COLUMNS].copy()
    y_text = df[TARGET_COLUMN].astype(str).copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    print("Encoded classes:")
    for idx, class_name in enumerate(label_encoder.classes_):
        print(f"  {idx} -> {class_name}")

    print("\n[3/7] Creating stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples       : {len(X_train)}")
    print(f"Test samples           : {len(X_test)}")

    # Save a true test-data CSV with the original text class label.
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = label_encoder.inverse_transform(y_test)
    test_data_path = PROJECT_ROOT / "test_data.csv"
    test_data.to_csv(test_data_path, index=False)
    print(f"Saved test data        : {test_data_path.name}")

    print("\n[4/7] Saving label encoder...")
    encoder_path = PROJECT_ROOT / "model" / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"Saved                  : {encoder_path.relative_to(PROJECT_ROOT)}")

    print("\n[5/7] Training and evaluating models...")
    models = get_models()
    results = []

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if not hasattr(model, "predict_proba"):
            raise RuntimeError(
                f"{model_name} does not provide predict_proba(), "
                "which is required for AUC."
            )

        y_proba = model.predict_proba(X_test)

        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            n_classes=len(label_encoder.classes_),
        )

        model_path = PROJECT_ROOT / "model" / MODEL_FILENAMES[model_name]
        save_model(model, model_path)

        result = {"ML Model Name": model_name, **metrics}
        results.append(result)

        for metric_name, value in metrics.items():
            print(f"{metric_name:10s}: {value:.4f}")

        print(f"Saved model: {model_path.relative_to(PROJECT_ROOT)}")

    print("\n[6/7] Creating comparison table...")
    results_df = pd.DataFrame(results)
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    results_df[metric_columns] = results_df[metric_columns].round(4)

    comparison_path = PROJECT_ROOT / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\n" + results_df.to_string(index=False))
    print(f"\nSaved comparison table : {comparison_path.name}")

    print("\n[7/7] Selecting overall winner...")
    # F1 is used as the primary ranking criterion; Accuracy breaks a tie.
    winner_df = results_df.sort_values(
        by=["F1", "Accuracy", "MCC"],
        ascending=False,
    )
    winner = winner_df.iloc[0]["ML Model Name"]
    print(f"Overall winner by weighted F1: {winner}")

    print("\nTraining completed successfully.")
    print("\nNext command:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
