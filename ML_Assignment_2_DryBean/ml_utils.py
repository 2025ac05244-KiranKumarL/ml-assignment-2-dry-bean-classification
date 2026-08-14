from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",   # original UCI spelling in the dataset
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

# Aliases are handled because some CSV/XLSX copies use slightly different spellings/case.
COLUMN_ALIASES = {
    "AspectRatio": "AspectRation",
    "Roundness": "roundness",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known Dry Bean column-name variations."""
    df = df.copy()
    rename_map = {}
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _read_local_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported dataset file type: {suffix}")

    return normalize_columns(df)


def load_dry_bean_dataset(project_root: str | Path = ".") -> pd.DataFrame:
    """
    Load the Dry Bean dataset.

    Search order:
    1) data/Dry_Bean_Dataset.csv
    2) data/Dry_Bean_Dataset.xlsx
    3) Fetch UCI dataset id=602 using ucimlrepo and save a local CSV copy.
    """
    root = Path(project_root).resolve()
    candidates = [
        root / "data" / "Dry_Bean_Dataset.csv",
        root / "data" / "Dry_Bean_Dataset.xlsx",
        root / "Dry_Bean_Dataset.csv",
        root / "Dry_Bean_Dataset.xlsx",
    ]

    for path in candidates:
        if path.exists():
            df = _read_local_dataset(path)
            validate_dataset(df)
            return df

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Dataset file was not found and ucimlrepo is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    try:
        dry_bean = fetch_ucirepo(id=602)
        X = dry_bean.data.features.copy()
        y = dry_bean.data.targets.copy()

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Unexpected target shape returned by UCI.")
            y = y.iloc[:, 0]

        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        if df.columns[-1] != TARGET_COLUMN:
            df = df.rename(columns={df.columns[-1]: TARGET_COLUMN})

        df = normalize_columns(df)
        validate_dataset(df)

        save_path = root / "data" / "Dry_Bean_Dataset.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        return df

    except Exception as exc:
        raise RuntimeError(
            "Could not load the Dry Bean dataset automatically. "
            "Check internet access, or manually place Dry_Bean_Dataset.csv/xlsx "
            "inside the data/ folder."
        ) from exc


def validate_dataset(df: pd.DataFrame) -> None:
    df = normalize_columns(df)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required column(s): " + ", ".join(missing)
        )

    if len(df) < 500:
        raise ValueError("Dataset must contain at least 500 instances.")

    if len(FEATURE_COLUMNS) < 12:
        raise ValueError("Dataset must contain at least 12 input features.")

    if df[required].isnull().any().any():
        warnings.warn(
            "Missing values were detected. Rows containing missing required values "
            "will need to be handled before training.",
            RuntimeWarning,
        )


def get_models() -> dict[str, object]:
    """Return the five models explicitly named in the assignment PDF."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "kNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def calculate_metrics(
    y_true,
    y_pred,
    y_proba,
    n_classes: int,
) -> dict[str, float]:
    """
    Multiclass metrics.

    Precision, Recall and F1 use weighted averaging.
    AUC uses One-vs-Rest (OvR) with weighted averaging.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
    except Exception:
        auc = np.nan

    metrics["AUC"] = auc

    return {
        "Accuracy": metrics["Accuracy"],
        "AUC": metrics["AUC"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "MCC": metrics["MCC"],
    }


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)
