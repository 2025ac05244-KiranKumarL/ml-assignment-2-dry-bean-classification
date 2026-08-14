# Dry Bean Classification - Machine Learning Assignment 2

## Problem Statement

The objective of this project is to classify dry bean grains into their bean
varieties using morphological measurements extracted from bean images.

The project trains multiple machine-learning classification models and provides
an interactive Streamlit application for evaluating the trained models on test
data.

## Dataset Description

Dataset: Dry Bean Dataset  
Source: UCI Machine Learning Repository  
UCI Dataset ID: 602

The project expects the Dry Bean dataset containing 16 input features and the
target column `Class`.

The training script first checks for:

- `data/Dry_Bean_Dataset.csv`
- `data/Dry_Bean_Dataset.xlsx`

If neither file exists, it attempts to fetch UCI dataset ID 602 through the
`ucimlrepo` package and stores a CSV copy inside the `data/` folder.

## Models Used

The assignment PDF explicitly names the following five models:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

> Note: The assignment text also says "all 6 ML models", but the PDF lists only
> five model names and the supplied comparison table contains the same five.
> This project therefore implements the five models explicitly named in the PDF.
> Add a sixth model only if your instructor issues an official clarification.

## Evaluation Metrics

For every model the following metrics are calculated:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

For the multiclass Dry Bean problem:

- Precision, Recall and F1 use weighted averaging.
- AUC uses weighted One-vs-Rest (OvR).

## Project Structure

```text
ML_Assignment_2_DryBean/
│
├── app.py
├── ml_utils.py
├── requirements.txt
├── README.md
├── test_data.csv                 # generated after training
├── model_comparison.csv          # generated after training
├── ML_Assignment_2.ipynb
│
├── data/
│   └── Dry_Bean_Dataset.csv      # generated/downloaded or manually placed
│
└── model/
    ├── __init__.py
    ├── train_models.py
    ├── logistic_regression.pkl   # generated
    ├── decision_tree.pkl         # generated
    ├── knn.pkl                   # generated
    ├── naive_bayes.pkl           # generated
    ├── random_forest.pkl         # generated
    └── label_encoder.pkl         # generated
```

## Local Setup

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train all models

```bash
python model/train_models.py
```

This command will:

- load/fetch the dataset,
- create an 80/20 stratified split,
- train all five models,
- calculate all six required metrics,
- save the model files,
- create `test_data.csv`,
- create `model_comparison.csv`.

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Streamlit Features

The app includes:

- CSV test-data upload
- Model-selection dropdown
- Predictions
- Accuracy
- AUC
- Precision
- Recall
- F1
- MCC
- Confusion matrix
- Classification report
- All-model comparison table
- Prediction CSV download

## GitHub Repository Link

Add your GitHub repository URL here after uploading the project.

`YOUR_GITHUB_REPOSITORY_LINK`

## Live Streamlit App Link

Add your Streamlit Community Cloud URL here after deployment.

`YOUR_STREAMLIT_APP_LINK`

## Model Comparison Table

Run `python model/train_models.py` and copy the values from
`model_comparison.csv` into this section before final submission.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |

## Model Performance Observations

Fill this section after running the models on BITS Virtual Lab. The observations
must be based on your actual metric values rather than generic statements.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| kNN | |
| Naive Bayes | |
| Random Forest (Ensemble) | |
| Overall Winner for your dataset? | |

## BITS Virtual Lab Screenshot

Perform the assignment on the BITS Virtual Lab and insert the required execution
screenshot into the final submitted PDF.
