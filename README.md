# 🛍️ Customer Analytics & Sales Prediction

An end-to-end Data Science project analyzing 500K+ real-world retail transactions to uncover customer behavior, predict sales, and identify churn risk using Machine Learning.

🔗 **Live Demo:** https://aman-customer-analytics.streamlit.app
📂 **GitHub:** https://github.com/amanwork5454-debug/CUSTOMER-ANALYTICS

---

## 📌 Business Problem

An online UK retailer needed answers to three key questions:
1. **Which customers are most valuable, and who is at risk of leaving?**
2. **Can we predict future invoice sales from time and order features?**
3. **How can we proactively reduce customer churn?**

This project addresses all three using real transaction data from the [UCI Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/online+retail).

---

## 🎯 Project Highlights

- 📦 **Real-world dataset**: UCI Online Retail UK (541K rows → 397K after cleaning)
- 👥 **Customer Segmentation**: RFM Analysis + K-Means Clustering → High / Medium / Low Value segments
- 🤖 **Sales Prediction**: 3 regression models compared, selected by cross-validated R²
- 🚨 **Churn Prediction**: Binary classifier on RFM features with ROC-AUC evaluation
- 📊 **Interactive Streamlit Dashboard**: 4 pages deployed on Streamlit Cloud

---

## 🔬 Sales Prediction — ML Model Comparison

Models were evaluated on both held-out test R² and 5-fold cross-validated R² to detect overfitting.
**Best model selected by CV R²** (more robust than test-set R²).

| Model | Test R² | MAE | CV R² (5-fold) |
|---|---|---|---|
| Linear Regression | 0.83 | £187 | 0.65 |
| Random Forest | 0.56 | £181 | **0.60** ← best CV R², selected |
| Gradient Boosting | 0.68 | £185 | 0.45 |

> 💡 **Why CV R²?** Linear Regression shows a test R² of 0.83 vs CV R² of 0.65 — a 0.18 gap indicating the test split was not representative of the full data distribution. CV R² gives a more honest estimate of generalization performance.

---

## 🚨 Churn Prediction — Classification

- **Churn definition**: Customers with no purchase in the last **90 days** (Recency > 90)
- **Churn rate in dataset**: ~33%
- **Models trained**: Logistic Regression, Random Forest

| Model | Accuracy | ROC-AUC | CV AUC |
|---|---|---|---|
| Logistic Regression | ~88% | ~0.95 | ~0.94 |
| Random Forest | ~89% | ~0.97 | ~0.96 |

Key finding: **Recency is by far the most important churn predictor** (feature importance ~80%).

---

## 🧠 Features Used

**Sales Prediction**
- Year, Month, Quarter, Day of Week, Day of Month, Is Weekend
- Number of Items per Invoice, Number of Distinct Products

**Churn Prediction**
- Recency (days since last purchase)
- Frequency (number of orders)
- Monetary (total lifetime spend)

---

## 📊 Dashboard Pages

1. **Overview** — KPI metrics, interactive monthly revenue trend, top products, revenue by country (Plotly)
2. **Customer Segments** — RFM scatter plot, segment breakdown (High / Medium / Low Value), data table
3. **Sales Prediction** — Model comparison, feature importance, live invoice sales predictor
4. **Churn Analysis** — Confusion matrix, ROC curve, feature importance, live churn risk predictor

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn (LinearRegression, RandomForest, GradientBoosting, LogisticRegression) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
customer-analytics/
├── app.py                      # Streamlit dashboard (4 pages)
├── model.pkl                   # Best sales prediction model + metrics
├── requirements.txt
├── data/
│   ├── cleaned_retail_sample.csv
│   ├── rfm.csv
│   ├── model_comparison.csv
│   └── high_risk_customers.csv
└── notebooks/
    ├── 01_cleaning.py           # Data cleaning pipeline
    ├── 02_eda.py                # Exploratory data analysis
    ├── 03_segmentation.py       # RFM + K-Means clustering
    ├── 04_prediction.py         # Sales regression models
    ├── 05_churn.py              # Churn classification models
    └── *.png                    # Generated charts
```

---

## 🔑 Key Learnings

- **RFM segmentation** is a simple but powerful framework for customer value analysis
- **Cross-validation is essential** — test-set R² alone can be misleading (seen in LR's 0.83 vs 0.65 CV gap)
- **Churn is highly predictable from recency alone** — a high recency value is a strong signal
- **Streamlit** enables rapid prototyping of ML dashboards without frontend expertise

---

## 👤 Author

**Aman Pokhriyal**
- GitHub: https://github.com/amanwork5454-debug
