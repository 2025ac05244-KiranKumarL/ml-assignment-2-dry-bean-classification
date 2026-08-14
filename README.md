# Dry Bean Classification — Machine Learning Assignment 2

## 1. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models for identifying different varieties of dry beans based on their morphological characteristics.

The project also provides an interactive Streamlit web application where test data can be uploaded, different trained models can be selected, predictions can be generated, and evaluation metrics can be viewed.

---

## 2. Dataset Description

**Dataset:** Dry Bean Dataset
**Source:** UCI Machine Learning Repository
**Problem Type:** Multiclass Classification

The dataset contains:

* **13,611 instances**
* **16 input features**
* **7 bean classes**
* **Target column:** `Class`

### Bean Classes

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

### Input Features

The dataset contains the following 16 numerical features:

* Area
* Perimeter
* MajorAxisLength
* MinorAxisLength
* AspectRation
* Eccentricity
* ConvexArea
* EquivDiameter
* Extent
* Solidity
* roundness
* Compactness
* ShapeFactor1
* ShapeFactor2
* ShapeFactor3
* ShapeFactor4

---

## 3. GitHub Repository Link

**GitHub Repository:**

https://github.com/2025ac05244-KiranKumarL/ml-assignment-2-dry-bean-classification

---

## 4. Live Streamlit Application

**Streamlit Application:**

https://ml-assignment-2-dry-bean-classification-c66mk3zrpsb8uutdzs8urk.streamlit.app/

The deployed application allows users to:

* Upload test data in CSV format
* Select a machine learning model
* Generate predictions
* View evaluation metrics
* View a confusion matrix
* View a classification report
* Compare all trained models
* Download prediction results

---

## 5. Machine Learning Models Used

The following classification models were implemented on the same Dry Bean dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

The dataset was divided into training and testing data using an **80:20 stratified train-test split** with a fixed random state for reproducibility.

Standardization was applied where required, particularly for Logistic Regression, kNN and Gaussian Naive Bayes.

---

## 6. Evaluation Metrics

Each model was evaluated using the following metrics:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

Since this is a multiclass classification problem:

* Precision, Recall and F1 Score are calculated using **weighted averaging**.
* AUC is calculated using the **weighted One-vs-Rest (OvR)** approach.

---

## 7. Model Comparison

| ML Model Name       |   Accuracy |        AUC |  Precision |     Recall |         F1 |        MCC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.9207 | **0.9934** |     0.9215 |     0.9207 |     0.9209 |     0.9042 |
| Decision Tree       |     0.8917 |     0.9330 |     0.8912 |     0.8917 |     0.8913 |     0.8691 |
| kNN                 |     0.9166 |     0.9812 |     0.9174 |     0.9166 |     0.9168 |     0.8992 |
| Naive Bayes         |     0.8979 |     0.9902 |     0.9007 |     0.8979 |     0.8981 |     0.8773 |
| **Random Forest**   | **0.9232** |     0.9921 | **0.9233** | **0.9232** | **0.9232** | **0.9072** |

---

## 8. Model Performance Observations

### Logistic Regression

Logistic Regression achieved an accuracy of **92.07%** and obtained the highest AUC score of **0.9934** among the tested models.

The model provided strong overall classification performance and showed very good class-separation capability.

The confusion matrix showed excellent classification for some classes such as BOMBAY, while some confusion was observed between similar classes such as DERMASON and SIRA.

---

### Decision Tree

Decision Tree achieved an accuracy of **89.17%**.

It produced the lowest overall performance among the tested models in terms of Accuracy, AUC, F1 and MCC.

A single decision tree may create highly specific decision boundaries based on the training data, which can reduce its ability to generalize compared with ensemble-based approaches.

---

### k-Nearest Neighbors

kNN achieved an accuracy of **91.66%** with an F1 score of **0.9168**.

The numerical features were standardized before applying kNN because distance-based algorithms are affected by differences in feature scales.

The model performed competitively but remained slightly below Logistic Regression and Random Forest.

---

### Naive Bayes

Gaussian Naive Bayes achieved an accuracy of **89.79%**.

Although its classification accuracy was lower than Logistic Regression, kNN and Random Forest, it achieved a very high AUC score of **0.9902**.

The model therefore showed good probability-based class-separation capability but was comparatively weaker in final class prediction.

---

### Random Forest

Random Forest achieved the best overall performance among the tested models.

It obtained:

* **Accuracy:** 0.9232
* **Precision:** 0.9233
* **Recall:** 0.9232
* **F1 Score:** 0.9232
* **MCC:** 0.9072
* **AUC:** 0.9921

Random Forest combines the predictions of multiple decision trees, which helps improve generalization and reduces the weaknesses associated with using a single decision tree.

---

## 9. Overall Winner

### 🏆 Random Forest

**Random Forest was selected as the overall best-performing model for the Dry Bean dataset.**

It achieved the highest:

* Accuracy
* Precision
* Recall
* F1 Score
* MCC Score

Although Logistic Regression achieved a slightly higher AUC score, Random Forest performed better across most of the required evaluation metrics and was therefore selected as the overall winner.

---

## 10. Streamlit Application Features

The Streamlit application provides an interactive interface for testing the trained machine learning models.

### Features

* Test-data CSV upload
* Machine learning model selection
* Dataset preview
* Prediction generation
* Actual vs predicted class comparison
* Accuracy display
* AUC display
* Precision display
* Recall display
* F1 Score display
* MCC Score display
* Confusion matrix
* Classification report
* All-model comparison table
* Best-model identification
* Model-performance comparison graph
* Prediction CSV download

---

## 11. Project Structure

```text
ml-assignment-2-dry-bean-classification/
│
├── app.py
├── ml_utils.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
├── ML_Assignment_2.ipynb
├── PROJECT_NOTES.txt
├── run_train.bat
├── run_streamlit.bat
│
├── data/
│   └── Dry_Bean_Dataset.csv
│
└── model/
    ├── train_models.py
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── label_encoder.pkl
```

---

## 12. Installation and Local Execution

### Clone the Repository

```bash
git clone https://github.com/2025ac05244-KiranKumarL/ml-assignment-2-dry-bean-classification.git
```

Move into the project directory:

```bash
cd ml-assignment-2-dry-bean-classification
```

### Create Virtual Environment

```bash
python -m venv venv
```

For Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Models

```bash
python model/train_models.py
```

The training process generates:

* Trained model files
* Label encoder
* `test_data.csv`
* `model_comparison.csv`

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will normally be available locally at:

```text
http://localhost:8501
```

---

## 13. Deployment

The application is deployed using **Streamlit Community Cloud**.

Repository branch:

```text
main
```

Main application file:

```text
app.py
```

Live Application:

https://ml-assignment-2-dry-bean-classification-c66mk3zrpsb8uutdzs8urk.streamlit.app/

---

## 14. Conclusion

Five machine learning classification algorithms were implemented and compared using the Dry Bean dataset.

All models produced reasonable classification performance, with Logistic Regression, kNN and Random Forest providing particularly strong results.

Random Forest achieved the strongest overall performance, reaching approximately **92.32% classification accuracy** and the highest F1 and MCC values among the tested models.

The completed Streamlit application demonstrates the models interactively and provides model selection, predictions, evaluation metrics, confusion matrices, classification reports and model-performance comparison.
