# 💳 Fraud Detection on Bank Transaction Data

This project explores and models fraudulent transactions from a bank payment dataset using various machine learning models. We focus on **Exploratory Data Analysis (EDA)**, **data preprocessing**, **imbalanced data handling using SMOTE**, and evaluating models like **K-Nearest Neighbors**, **Random Forest**, and **XGBoost**.

---

## 📂 Dataset

The dataset used is `bs140513_032310.csv`, which includes over 594,000 records of transactions with features like:

- `step`: Hour of transaction  
- `customer`, `merchant`: Categorical IDs  
- `age`, `gender`  
- `category`: Merchant category  
- `amount`: Transaction amount  
- `fraud`: Target label (0 = Non-fraud, 1 = Fraud)

---

## 📊 Exploratory Data Analysis (EDA)

EDA reveals:

- Highly imbalanced classes (only 7200 fraudulent out of ~594k)
- Fraudulent transactions tend to have **higher amounts**
- Categories like **leisure**, **travel**, and **sports** have the highest fraud rates
- Age group `'0'` (possibly under 18 or invalid) shows **higher fraud percent**

### Visualizations:

- Count plots, box plots, and histograms were used to show distributions.
- ROC-AUC curves were plotted for model performance.

---

## 🧹 Data Preprocessing

Steps included:

- Dropping low-informative features (`zipcodeOri`, `zipMerchant`)
- Converting object types to categorical codes
- Splitting features `X` and target `y`
- Applying **SMOTE** to handle class imbalance

---

## ⚙️ Models Used

We trained and evaluated three models:

### 1. K-Nearest Neighbors (KNN)
- Very high accuracy (~99%)  
- Performed well after applying SMOTE  
- Fast and simple for baseline comparison

### 2. Random Forest
- Handled class imbalance using `class_weight="balanced"`  
- Accuracy: ~98%  
- Robust and interpretable

### 3. XGBoost
- Configured with `binary:hinge` objective  
- Handled large data efficiently  
- Competitive performance

---

## 📈 Performance Metrics

All models were evaluated using:

- Confusion Matrix  
- Classification Report (precision, recall, f1-score)  
- ROC-AUC Curves

> ⚠️ Without SMOTE, the models struggled due to heavy class imbalance.  
> SMOTE significantly improved the performance on minority (fraud) class.

---

## 📌 Conclusion

This project shows that:

- **SMOTE is essential** when dealing with imbalanced fraud data.
- **XGBoost** and **Random Forest** provide strong results on large datasets.
- **Fraudulent behavior can be understood** using EDA before applying ML models.

---

## 📁 Project Structure

