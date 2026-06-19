# SMS Spam Detection using Machine Learning

## Project Overview

SMS Spam Detection is a Natural Language Processing (NLP) and Machine Learning project that classifies SMS messages as either **Spam** or **Ham (Legitimate Message)**. Spam messages are unwanted promotional or fraudulent messages that can mislead users, while ham messages are normal messages exchanged between users.

The objective of this project is to build an efficient machine learning model capable of automatically identifying spam messages based on the content of the SMS.

---

## Dataset Information

**Dataset Name:** SMS Spam Collection Dataset

**Total Records:** 5,572 SMS Messages

### Dataset Features

| Feature | Description                    |
| ------- | ------------------------------ |
| Label   | Message category (Ham or Spam) |
| Message | SMS text content               |

### Class Distribution

| Class | Count |
| ----- | ----- |
| Ham   | 4825  |
| Spam  | 747   |

The dataset is slightly imbalanced, with ham messages significantly outnumbering spam messages.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Natural Language Processing (NLP)

---

## Project Workflow

### 1. Data Collection

The SMS Spam Collection Dataset was loaded into a Pandas DataFrame for analysis and processing.

### 2. Data Cleaning

The following preprocessing steps were performed:

* Removed unnecessary columns:

  * Unnamed: 2
  * Unnamed: 3
  * Unnamed: 4
* Renamed columns:

  * v1 → Label
  * v2 → Message
* Checked for missing values.
* Verified dataset structure and data types.

### 3. Exploratory Data Analysis (EDA)

Several analyses were performed:

#### Class Distribution Analysis

* Counted the number of spam and ham messages.
* Visualized message categories using Count Plot.

#### Message Length Analysis

A new feature called **Length** was created using:

```python
df['Length'] = df['Message'].apply(len)
```

This feature helped analyze differences between spam and ham message lengths.

### Length Statistics

| Metric         | Value |
| -------------- | ----- |
| Mean Length    | 80.12 |
| Minimum Length | 2     |
| Maximum Length | 910   |

Average Message Length by Class:

| Class | Average Length |
| ----- | -------------- |
| Ham   | 71.02          |
| Spam  | 138.87         |

Observation:
Spam messages are generally much longer than ham messages.

---

## Data Preprocessing

### Label Encoding

The target labels were converted into numerical values:

| Original Label | Encoded Value |
| -------------- | ------------- |
| Ham            | 0             |
| Spam           | 1             |

---

## Feature Extraction using TF-IDF

Text data cannot be directly processed by machine learning algorithms.

Therefore, messages were converted into numerical feature vectors using **TF-IDF (Term Frequency - Inverse Document Frequency)**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Message'])
```

### TF-IDF Output

Shape:

```text
(5572, 8672)
```

Meaning:

* 5572 SMS messages
* 8672 unique vocabulary features

---

## Train-Test Split

The dataset was divided into:

* Training Set: 80%
* Testing Set: 20%

```python
X_train.shape = (4457, 8672)
X_test.shape  = (1115, 8672)
```

---

# Machine Learning Models

## 1. Multinomial Naive Bayes

Naive Bayes is a probabilistic classification algorithm commonly used in text classification tasks.

### Results

Accuracy:

```text
96.23%
```

Confusion Matrix:

```text
[[965   0]
 [ 42 108]]
```

---

## 2. Logistic Regression

Logistic Regression is a supervised classification algorithm that predicts probabilities using the sigmoid function.

### Results

Accuracy:

```text
96.32%
```

Confusion Matrix:

```text
[[965   0]
 [ 41 109]]
```

---

## 3. Support Vector Machine (SVM)

Support Vector Machine finds the optimal decision boundary separating spam and ham messages.

### Results

Accuracy:

```text
98.03%
```

Confusion Matrix:

```text
[[963   2]
 [ 20 130]]
```

---

# Model Performance Comparison

| Model                        | Accuracy |
| ---------------------------- | -------- |
| Naive Bayes                  | 96.23%   |
| Logistic Regression          | 96.32%   |
| Support Vector Machine (SVM) | 98.03%   |

### Best Model

**Support Vector Machine (SVM)** achieved the highest performance among all tested models.

Reasons:

* Highest Accuracy
* Highest Recall
* Highest F1-Score
* Better Spam Detection Capability

---

# Evaluation Metrics Used

### Accuracy

Measures the percentage of correctly classified messages.

### Precision

Measures how many predicted spam messages were actually spam.

### Recall

Measures how many actual spam messages were correctly identified.

### F1 Score

Balances precision and recall into a single metric.

### Confusion Matrix

Provides detailed information about correct and incorrect predictions.

---

# Results

* Successfully built an SMS Spam Detection System.
* Applied NLP techniques for text preprocessing.
* Converted SMS text into numerical features using TF-IDF.
* Trained and evaluated multiple machine learning models.
* Achieved a maximum accuracy of **98.03%** using Support Vector Machine.
* Demonstrated effective spam message classification.

---

# Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be combined to build an efficient SMS Spam Detection System. After preprocessing and feature extraction using TF-IDF, three machine learning algorithms were evaluated. Among them, Support Vector Machine (SVM) achieved the best performance with an accuracy of 98.03%.

The developed system can effectively classify incoming SMS messages as Spam or Ham, helping users reduce unwanted and potentially harmful messages.
