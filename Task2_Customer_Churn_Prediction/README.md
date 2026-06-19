# Customer Churn Prediction using Machine Learning

## Project Overview

Customer churn refers to customers leaving a company's service or discontinuing their relationship with the organization. Predicting customer churn helps businesses identify customers who are likely to leave and take preventive actions to improve customer retention.

This project uses Machine Learning techniques to predict whether a bank customer will leave the bank based on customer demographic and financial information.

---

## Problem Statement

The objective of this project is to build a machine learning model capable of predicting customer churn using historical customer data.

The target variable is:

* **Exited = 0** → Customer stays with the bank
* **Exited = 1** → Customer leaves the bank

By accurately identifying customers who are likely to leave, organizations can implement targeted retention strategies and reduce customer loss.

---

## Dataset Information

Dataset Name:

**Churn_Modelling.csv**

Dataset Size:

* Total Records: **10,000**
* Total Features: **14**

### Features

| Feature         | Description              |
| --------------- | ------------------------ |
| RowNumber       | Record Index             |
| CustomerId      | Unique Customer ID       |
| Surname         | Customer Last Name       |
| CreditScore     | Customer Credit Score    |
| Geography       | Customer Country         |
| Gender          | Male/Female              |
| Age             | Customer Age             |
| Tenure          | Years with Bank          |
| Balance         | Account Balance          |
| NumOfProducts   | Number of Bank Products  |
| HasCrCard       | Credit Card Ownership    |
| IsActiveMember  | Active Membership Status |
| EstimatedSalary | Estimated Salary         |
| Exited          | Target Variable          |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Jupyter Notebook

---

## Project Workflow

### 1. Data Loading

The dataset was loaded using Pandas and initial exploration was performed.

```python
df = pd.read_csv("Churn_Modelling.csv")
```

---

### 2. Data Exploration

The following operations were performed:

* Dataset shape inspection
* Data type analysis
* Statistical summary
* Missing value detection
* Duplicate record detection

Results:

* Missing Values: 0
* Duplicate Records: 0

---

### 3. Target Variable Analysis

Target variable:

```text
Exited
```

Distribution:

| Class       | Count |
| ----------- | ----- |
| Stayed (0)  | 7963  |
| Churned (1) | 2037  |

Percentage:

* Stayed: 79.63%
* Churned: 20.37%

The dataset is moderately imbalanced.

---

### 4. Exploratory Analysis

#### Geography vs Churn

Analysis showed that customers from Germany had significantly higher churn rates compared to France and Spain.

#### Gender vs Churn

Female customers exhibited higher churn rates than male customers.

---

### 5. Data Preprocessing

#### Removed Columns

The following columns were removed because they do not contribute to churn prediction:

```text
RowNumber
CustomerId
Surname
```

#### Encoding

##### Label Encoding

Applied to:

```text
Gender
```

Conversion:

```text
Male   → 1
Female → 0
```

##### One-Hot Encoding

Applied to:

```text
Geography
```

Generated:

```text
Geography_Germany
Geography_Spain
```

France was used as the reference category.

---

### 6. Feature and Target Separation

```python
X = df.drop("Exited", axis=1)
y = df["Exited"]
```

---

### 7. Train-Test Split

Dataset split:

* Training Data: 80%
* Testing Data: 20%

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 8. Feature Scaling

StandardScaler was applied for Logistic Regression.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression predicts the probability of customer churn.

#### Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 81.10% |
| Precision | 0.55   |
| Recall    | 0.20   |
| F1 Score  | 0.29   |

Observation:

The model achieved reasonable accuracy but struggled to identify churned customers effectively.

---

### 2. Decision Tree Classifier

Decision Trees create rule-based classifications by splitting data into decision nodes.

#### Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 78.45% |
| Precision | 0.46   |
| Recall    | 0.50   |
| F1 Score  | 0.48   |

Observation:

Decision Tree significantly improved recall compared to Logistic Regression.

---

### 3. Random Forest Classifier

Random Forest combines multiple Decision Trees and uses majority voting for predictions.

#### Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 86.75% |
| Precision | 0.76   |
| Recall    | 0.47   |
| F1 Score  | 0.58   |

Observation:

Random Forest produced the best overall performance.

---

## Model Comparison

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 81.10%   |
| Decision Tree       | 78.45%   |
| Random Forest       | 86.75%   |

### Best Model

🏆 **Random Forest Classifier**

Reasons:

* Highest Accuracy
* Highest Precision
* Highest F1 Score
* Good Recall Performance

---

## Feature Importance Analysis

Random Forest Feature Importance:

| Feature           | Importance |
| ----------------- | ---------- |
| Age               | 0.238      |
| EstimatedSalary   | 0.147      |
| CreditScore       | 0.143      |
| Balance           | 0.139      |
| NumOfProducts     | 0.131      |
| Tenure            | 0.082      |
| IsActiveMember    | 0.042      |
| Geography_Germany | 0.026      |
| Gender            | 0.019      |
| HasCrCard         | 0.019      |
| Geography_Spain   | 0.014      |

### Key Factors Influencing Churn

* Age
* Estimated Salary
* Credit Score
* Balance
* Number of Products
* Tenure

---

## Conclusion

This project successfully developed a Customer Churn Prediction system using Machine Learning techniques.

Three classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

Among all models, Random Forest achieved the best performance with an accuracy of **86.75%** and demonstrated superior predictive capability.

The project highlights how machine learning can assist organizations in identifying customers who are likely to leave, enabling proactive retention strategies and improved customer relationship management.

---

## Future Improvements

* Hyperparameter Tuning
* Cross Validation
* Feature Selection
* SMOTE for Class Imbalance Handling
* XGBoost and LightGBM Implementation
* Deployment using Flask or Streamlit

---

## Author

**Manikanta Amarapalli**

B.Tech CSE Student
RGUKT Nuzvid
