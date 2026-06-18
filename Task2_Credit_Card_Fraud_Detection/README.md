# Credit Card Fraud Detection using Machine Learning

## Project Overview

Credit card fraud has become a major challenge in the digital payment ecosystem. Financial institutions process millions of transactions every day, making it difficult to manually identify fraudulent activities. This project aims to build and evaluate machine learning models capable of distinguishing between legitimate and fraudulent credit card transactions.

The project explores data preprocessing, feature engineering, handling class imbalance, and comparing multiple machine learning algorithms for fraud detection.

---

## Problem Statement

The objective of this project is to develop a machine learning model that can classify a credit card transaction as:

* **0 → Legitimate Transaction**
* **1 → Fraudulent Transaction**

Since fraudulent transactions represent only a very small portion of the dataset, this is a highly imbalanced classification problem.

---

## Dataset Information

The dataset consists of two files:

* `fraudTrain.csv`
* `fraudTest.csv`

### Dataset Size

| Dataset       | Records   |
| ------------- | --------- |
| Training Data | 1,296,675 |
| Testing Data  | 555,719   |

### Target Variable

| Value | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

### Class Distribution

| Class      | Percentage |
| ---------- | ---------- |
| Legitimate | 99.42%     |
| Fraudulent | 0.58%      |

This severe imbalance makes fraud detection particularly challenging.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Project Workflow

### 1. Data Loading and Exploration

The dataset was loaded using Pandas and inspected using:

* `head()`
* `shape`
* `info()`
* `isnull()`

The dataset was found to contain no missing values.

---

### 2. Feature Engineering

Two new features were extracted:

#### Hour

Transaction timestamp was converted into datetime format and the transaction hour was extracted.

Example:

* 02:00 AM
* 03:00 PM

Fraudulent behavior often varies depending on transaction time.

#### Age

Customer age was calculated using:

* Transaction Date
* Date of Birth

This transformed the raw date of birth into a more meaningful feature.

---

### 3. Feature Selection

Columns that contained identifiers or irrelevant information were removed:

* Unnamed: 0
* first
* last
* street
* trans_num
* cc_num
* city
* merchant
* job
* dob
* trans_date_trans_time

These columns either contained unique identifiers or high-cardinality categorical values that were not suitable for the initial model.

---

### 4. Categorical Encoding

The following categorical columns were transformed using One-Hot Encoding:

* category
* gender
* state

This allowed machine learning algorithms to process categorical information numerically.

---

### 5. Feature Scaling

Since numerical features existed on different scales, StandardScaler was applied.

Examples:

* hour → 0–23
* age → 20–90
* unix_time → billions

Scaling ensured fair contribution from all features during model training.

---

## Machine Learning Models

### Logistic Regression

A baseline Logistic Regression model was trained.

#### Results

* Accuracy: 99.57%
* Precision: 0.04
* Recall: 0.004
* F1 Score: 0.01

Although the accuracy appeared extremely high, the model detected only 9 fraudulent transactions out of 2145 fraud cases.

This demonstrated why accuracy alone is not an appropriate metric for highly imbalanced datasets.

---

### Balanced Logistic Regression

To address class imbalance, Logistic Regression was trained using:

```python
class_weight='balanced'
```

#### Results

* Precision: 0.04
* Recall: 0.73
* F1 Score: 0.07

Recall improved significantly, but the model generated a large number of false positives.

---

### Decision Tree Classifier

A Decision Tree model was trained and evaluated.

#### Results

* Precision: 0.60
* Recall: 0.81
* F1 Score: 0.69

The Decision Tree successfully detected a large percentage of fraud cases and significantly outperformed Logistic Regression.

---

### Random Forest Classifier

Multiple Random Forest configurations were evaluated.

#### Best Results

* Precision: 0.97
* Recall: 0.65
* F1 Score: 0.77

The Random Forest model provided the best overall balance between fraud detection capability and false alarm reduction.

---

## Model Comparison

| Model                        | Precision | Recall | F1 Score |
| ---------------------------- | --------- | ------ | -------- |
| Logistic Regression          | 0.04      | 0.004  | 0.01     |
| Balanced Logistic Regression | 0.04      | 0.73   | 0.07     |
| Decision Tree                | 0.60      | 0.81   | 0.69     |
| Random Forest                | 0.97      | 0.65   | 0.77     |

---

## Key Learnings

This project highlights several important machine learning concepts:

* Data preprocessing and cleaning
* Feature engineering
* Handling categorical variables
* Feature scaling
* Class imbalance challenges
* Precision, Recall, and F1 Score evaluation
* Decision Trees and Random Forests
* Why accuracy is not always the best evaluation metric

---

## Conclusion

The dataset was highly imbalanced, containing only 0.58% fraudulent transactions.

While Logistic Regression achieved high accuracy, it failed to effectively identify fraudulent transactions. Balanced Logistic Regression improved fraud detection but introduced many false positives.

Decision Tree achieved the highest recall and detected the largest number of fraudulent transactions. Random Forest achieved the highest F1 Score and precision, making it the best overall model for this fraud detection task.

The results demonstrate that evaluation metrics such as Precision, Recall, and F1 Score provide a more reliable assessment than accuracy when dealing with imbalanced datasets.

---

## Author

**Manikanta Amarapalli**

Computer Science and Engineering Student
RGUKT Nuzvid
