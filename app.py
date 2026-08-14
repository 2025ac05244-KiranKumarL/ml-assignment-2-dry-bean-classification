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


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.35rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .sub-text {
            font-size: 1rem;
            opacity: 0.78;
            margin-bottom: 1.5rem;
        }

        .winner-box {
            border: 1px solid rgba(80, 200, 120, 0.45);
            border-radius: 12px;
            padding: 16px 18px;
            margin-top: 10px;
            margin-bottom: 18px;
            background: rgba(40, 120, 70, 0.12);
        }

        .section-note {
            font-size: 0.92rem;
            opacity: 0.75;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(150, 150, 150, 0.20);
            padding: 14px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🫘 Dry Bean Classification — ML Assignment 2</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-text">
        Interactive machine-learning dashboard for evaluating multiple
        classification models on the Dry Bean dataset.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET SUMMARY CARDS
# ============================================================

summary_cols = st.columns(4)

with summary_cols[0]:
    st.metric("Dataset Samples", "13,611")

with summary_cols[1]:
    st.metric("Input Features", "16")

with summary_cols[2]:
    st.metric("Bean Classes", "7")

with summary_cols[3]:
    st.metric("ML Models", "5")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Controls")

    selected_model_name = st.selectbox(
        "Select ML model",
        list(MODEL_FILES.keys()),
    )

    uploaded_file = st.file_uploader(
        "Upload test-data CSV",
        type=["csv"],
        help=(
            "Upload test data containing the 16 Dry Bean feature columns. "
            "Include the Class column to calculate evaluation metrics."
        ),
    )

    use_bundled_test = st.checkbox(
        "Use generated test_data.csv when no file is uploaded",
        value=True,
    )

    st.markdown("---")

    st.subheader("Dataset Information")
    st.write("**Dataset:** Dry Bean")
    st.write("**Problem:** Multiclass Classification")
    st.write("**Required features:** 16")
    st.write("**Target column:** `Class`")
    st.write("**Classes:** 7")


# ============================================================
# MODEL / ENCODER LOADING
# ============================================================

@st.cache_resource
def load_assets():
    encoder_path = MODEL_DIR / "label_encoder.pkl"

    required_paths = list(MODEL_FILES.values()) + [encoder_path]

    missing = [
        path.name
        for path in required_paths
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


# ============================================================
# TEST DATA LOADING
# ============================================================

def get_input_dataframe():
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "Uploaded CSV"

    bundled_test_path = PROJECT_ROOT / "test_data.csv"

    if use_bundled_test and bundled_test_path.exists():
        return pd.read_csv(bundled_test_path), "Generated test_data.csv"

    return None, None


models, label_encoder, missing_assets = load_assets()

if missing_assets:
    st.error(
        "Some trained model files are missing. "
        "Run the training script before starting Streamlit."
    )

    st.code("python model/train_models.py")

    st.write("Missing files:")
    for file_name in missing_assets:
        st.write(f"- {file_name}")

    st.stop()


df, source_name = get_input_dataframe()

if df is None:
    st.info(
        "Upload a CSV file using the sidebar, or enable the generated "
        "test_data.csv option."
    )
    st.stop()

df = normalize_columns(df)


# ============================================================
# 1. TEST DATA PREVIEW
# ============================================================

st.markdown("---")
st.header("1. Test Data Preview")

st.write(f"Data source: **{source_name}**")

preview_rows = st.slider(
    "Number of preview rows",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
)

st.dataframe(
    df.head(preview_rows),
    use_container_width=True,
)


# ============================================================
# VALIDATE INPUT FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in df.columns
]

if missing_features:
    st.error(
        "The uploaded CSV is missing required feature columns: "
        + ", ".join(missing_features)
    )
    st.stop()

X_test = df[FEATURE_COLUMNS].copy()

if X_test.isnull().any().any():
    st.error(
        "Missing values were found in one or more required feature columns. "
        "Please clean the test data and upload it again."
    )
    st.stop()


# ============================================================
# 2. MODEL SELECTION
# ============================================================

st.header("2. Selected Model")

st.success(f"Currently evaluating: {selected_model_name}")

model = models[selected_model_name]


# ============================================================
# 3. PREDICTIONS
# ============================================================

st.header("3. Prediction Results")

y_pred = model.predict(X_test)

predicted_labels = label_encoder.inverse_transform(y_pred)

prediction_df = df.copy()
prediction_df["Predicted_Class"] = predicted_labels

if TARGET_COLUMN in df.columns:
    prediction_df["Prediction_Status"] = np.where(
        prediction_df[TARGET_COLUMN].astype(str)
        == prediction_df["Predicted_Class"].astype(str),
        "Correct",
        "Incorrect",
    )

st.dataframe(
    prediction_df.head(50),
    use_container_width=True,
)

prediction_csv = prediction_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download predictions as CSV",
    data=prediction_csv,
    file_name="dry_bean_predictions.csv",
    mime="text/csv",
)


# ============================================================
# STOP IF TARGET IS NOT AVAILABLE
# ============================================================

if TARGET_COLUMN not in df.columns:
    st.warning(
        "The uploaded file does not contain the Class column. "
        "Predictions are displayed, but evaluation metrics cannot be calculated."
    )
    st.stop()


# ============================================================
# PREPARE TRUE LABELS
# ============================================================

try:
    y_true = label_encoder.transform(
        df[TARGET_COLUMN].astype(str)
    )
except ValueError as exc:
    st.error(
        "The Class column contains one or more labels that were not present "
        "during model training."
    )
    st.exception(exc)
    st.stop()


if not hasattr(model, "predict_proba"):
    st.error(
        "The selected model does not support probability prediction, "
        "so AUC cannot be calculated."
    )
    st.stop()

y_proba = model.predict_proba(X_test)


# ============================================================
# 4. EVALUATION METRICS
# ============================================================

metrics = calculate_metrics(
    y_true=y_true,
    y_pred=y_pred,
    y_proba=y_proba,
    n_classes=len(label_encoder.classes_),
)

st.header("4. Evaluation Metrics")

metric_names = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

row1 = st.columns(3)
row2 = st.columns(3)

for index, metric_name in enumerate(metric_names):
    value = metrics[metric_name]

    target_column = (
        row1[index]
        if index < 3
        else row2[index - 3]
    )

    with target_column:
        if np.isnan(value):
            st.metric(metric_name, "N/A")
        else:
            st.metric(metric_name, f"{value:.4f}")

st.markdown(
    """
    <div class="section-note">
        Precision, Recall and F1 are calculated using weighted averaging.
        Multiclass AUC is calculated using weighted One-vs-Rest (OvR).
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

st.header("5. Confusion Matrix")

labels_numeric = np.arange(
    len(label_encoder.classes_)
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels_numeric,
)

fig_cm, ax_cm = plt.subplots(figsize=(9, 6))

image = ax_cm.imshow(cm)
fig_cm.colorbar(image, ax=ax_cm)

ax_cm.set_xticks(labels_numeric)
ax_cm.set_yticks(labels_numeric)

ax_cm.set_xticklabels(
    label_encoder.classes_,
    rotation=45,
    ha="right",
)

ax_cm.set_yticklabels(
    label_encoder.classes_
)

ax_cm.set_xlabel("Predicted Class")
ax_cm.set_ylabel("Actual Class")

ax_cm.set_title(
    f"Confusion Matrix — {selected_model_name}"
)

threshold = cm.max() / 2 if cm.size else 0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),
        )

fig_cm.tight_layout()

st.pyplot(fig_cm)


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

st.header("6. Classification Report")

report = classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)

report_df = (
    pd.DataFrame(report)
    .transpose()
    .round(4)
)

st.dataframe(
    report_df,
    use_container_width=True,
)


# ============================================================
# 7. ALL-MODEL COMPARISON
# ============================================================

comparison_path = PROJECT_ROOT / "model_comparison.csv"

if comparison_path.exists():
    st.header("7. All-Model Comparison")

    comparison_df = pd.read_csv(comparison_path)

    st.dataframe(
        comparison_df,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if "F1" in comparison_df.columns:
        winner_row = (
            comparison_df
            .sort_values(
                by=["F1", "Accuracy", "MCC"],
                ascending=False,
            )
            .iloc[0]
        )

        winner_name = winner_row["ML Model Name"]
        winner_accuracy = winner_row["Accuracy"]
        winner_f1 = winner_row["F1"]
        winner_mcc = winner_row["MCC"]

        st.markdown(
            f"""
            <div class="winner-box">
                <h3>🏆 Best Overall Model: {winner_name}</h3>
                <p>
                    Accuracy: <b>{winner_accuracy:.4f}</b> &nbsp; | &nbsp;
                    F1 Score: <b>{winner_f1:.4f}</b> &nbsp; | &nbsp;
                    MCC: <b>{winner_mcc:.4f}</b>
                </p>
                <p>
                    The winner is selected primarily using weighted F1 score,
                    with Accuracy and MCC used as tie-breakers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # PERFORMANCE CHART
    # --------------------------------------------------------

    st.subheader("Model Performance Comparison")

    chart_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]

    available_chart_metrics = [
        metric
        for metric in chart_metrics
        if metric in comparison_df.columns
    ]

    selected_chart_metric = st.selectbox(
        "Choose metric for comparison chart",
        available_chart_metrics,
        index=0,
    )

    chart_data = comparison_df[
        ["ML Model Name", selected_chart_metric]
    ].copy()

    chart_data = chart_data.sort_values(
        selected_chart_metric,
        ascending=False,
    )

    fig_bar, ax_bar = plt.subplots(figsize=(9, 5))

    bars = ax_bar.bar(
        chart_data["ML Model Name"],
        chart_data[selected_chart_metric],
    )

    ax_bar.set_ylabel(selected_chart_metric)
    ax_bar.set_xlabel("ML Model")
    ax_bar.set_title(
        f"{selected_chart_metric} Comparison Across Models"
    )

    ax_bar.tick_params(
        axis="x",
        rotation=25,
    )

    maximum_value = chart_data[selected_chart_metric].max()

    for bar, value in zip(
        bars,
        chart_data[selected_chart_metric],
    ):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    ax_bar.set_ylim(
        0,
        min(1.08, maximum_value + 0.08),
    )

    fig_bar.tight_layout()

    st.pyplot(fig_bar)

else:
    st.info(
        "model_comparison.csv was not found. "
        "Run python model/train_models.py to generate it."
    )


# ============================================================
# 8. PROJECT OBSERVATION
# ============================================================

st.header("8. Project Observation")

st.write(
    """
    The evaluated models show that ensemble and linear methods perform strongly
    on the Dry Bean dataset. Random Forest provides the strongest overall
    classification performance across most evaluation metrics, while Logistic
    Regression achieves excellent probability-based class separation as shown
    by its AUC score.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Machine Learning Assignment 2 — Dry Bean Classification Dashboard"
)
