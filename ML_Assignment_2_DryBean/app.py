from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ml_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    calculate_metrics,
    normalize_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
}


st.set_page_config(
    page_title="Dry Bean ML Classifier",
    page_icon="🫘",
    layout="wide",
)

st.title("🫘 Dry Bean Classification — ML Assignment 2")
st.caption(
    "Compare multiple classification models on Dry Bean test data."
)

with st.sidebar:
    st.header("Controls")
    selected_model_name = st.selectbox(
        "Select ML model",
        list(MODEL_FILES.keys()),
    )

    uploaded_file = st.file_uploader(
        "Upload test-data CSV",
        type=["csv"],
        help=(
            "The CSV should contain the 16 Dry Bean feature columns. "
            "Include the Class column to calculate evaluation metrics."
        ),
    )

    use_bundled_test = st.checkbox(
        "Use generated test_data.csv when no file is uploaded",
        value=True,
    )

    st.markdown("---")
    st.write("**Required features:** 16")
    st.write("**Target column:** `Class`")


@st.cache_resource
def load_assets():
    encoder_path = MODEL_DIR / "label_encoder.pkl"

    missing = [
        str(path.name)
        for path in list(MODEL_FILES.values()) + [encoder_path]
        if not path.exists()
    ]

    if missing:
        return None, None, missing

    models = {
        name: joblib.load(path)
        for name, path in MODEL_FILES.items()
    }
    encoder = joblib.load(encoder_path)

    return models, encoder, []


def get_input_dataframe():
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "Uploaded CSV"

    bundled = PROJECT_ROOT / "test_data.csv"
    if use_bundled_test and bundled.exists():
        return pd.read_csv(bundled), "Generated test_data.csv"

    return None, None


models, label_encoder, missing_assets = load_assets()

if missing_assets:
    st.error(
        "Trained model files are missing. Run the training script first."
    )
    st.code("python model/train_models.py")
    st.write("Missing files:", ", ".join(missing_assets))
    st.stop()


df, source_name = get_input_dataframe()

if df is None:
    st.info(
        "Upload a CSV from the sidebar. "
        "You may also train the project first to generate test_data.csv."
    )
    st.stop()

df = normalize_columns(df)

st.subheader("1. Test Data Preview")
st.write(f"Data source: **{source_name}**")
st.dataframe(df.head(15), use_container_width=True)

missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing_features:
    st.error(
        "The uploaded CSV is missing required feature columns: "
        + ", ".join(missing_features)
    )
    st.stop()

X_test = df[FEATURE_COLUMNS].copy()

if X_test.isnull().any().any():
    st.error(
        "The test data contains missing values in required feature columns. "
        "Please remove/fill them and upload again."
    )
    st.stop()

model = models[selected_model_name]

st.subheader("2. Selected Model")
st.success(selected_model_name)

y_pred = model.predict(X_test)
predicted_labels = label_encoder.inverse_transform(y_pred)

prediction_df = df.copy()
prediction_df["Predicted_Class"] = predicted_labels

st.subheader("3. Prediction Results")
st.dataframe(
    prediction_df.head(50),
    use_container_width=True,
)

csv_bytes = prediction_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions as CSV",
    data=csv_bytes,
    file_name="dry_bean_predictions.csv",
    mime="text/csv",
)

if TARGET_COLUMN not in df.columns:
    st.warning(
        "The uploaded file has no Class column. Predictions are shown, "
        "but evaluation metrics cannot be calculated."
    )
    st.stop()


try:
    y_true = label_encoder.transform(df[TARGET_COLUMN].astype(str))
except ValueError as exc:
    st.error(
        "The Class column contains a label that was not seen during training."
    )
    st.exception(exc)
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error("Selected model does not support probability prediction.")
    st.stop()

y_proba = model.predict_proba(X_test)

metrics = calculate_metrics(
    y_true=y_true,
    y_pred=y_pred,
    y_proba=y_proba,
    n_classes=len(label_encoder.classes_),
)

st.subheader("4. Evaluation Metrics")

metric_order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metric_cols = st.columns(3)

for idx, metric_name in enumerate(metric_order):
    value = metrics[metric_name]
    with metric_cols[idx % 3]:
        if np.isnan(value):
            st.metric(metric_name, "N/A")
        else:
            st.metric(metric_name, f"{value:.4f}")

st.caption(
    "For this multiclass problem, Precision, Recall and F1 are weighted averages. "
    "AUC is weighted One-vs-Rest (OvR)."
)

st.subheader("5. Confusion Matrix")

labels_numeric = np.arange(len(label_encoder.classes_))
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels_numeric,
)

fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(cm)
fig.colorbar(image, ax=ax)

ax.set_xticks(labels_numeric)
ax.set_yticks(labels_numeric)
ax.set_xticklabels(label_encoder.classes_, rotation=45, ha="right")
ax.set_yticklabels(label_encoder.classes_)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(f"Confusion Matrix — {selected_model_name}")

threshold = cm.max() / 2 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
        )

fig.tight_layout()
st.pyplot(fig)

st.subheader("6. Classification Report")

report = classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(4), use_container_width=True)

comparison_path = PROJECT_ROOT / "model_comparison.csv"
if comparison_path.exists():
    st.subheader("7. All-Model Comparison")
    comparison_df = pd.read_csv(comparison_path)
    st.dataframe(comparison_df, use_container_width=True)

st.markdown("---")
st.caption(
    "Dry Bean classification project created for Machine Learning Assignment 2."
)
