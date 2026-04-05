# 🛍️ Customer Analytics & Sales Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-green)
![Prophet](https://img.shields.io/badge/Prophet-forecasting-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end Data Science project analyzing 500K+ real-world retail transactions to uncover customer behavior, predict sales, forecast future revenue, and identify churn risk using Machine Learning.

🔗 **Live Demo:** https://aman-customer-analytics.streamlit.app
📂 **GitHub:** https://github.com/amanwork5454-debug/CUSTOMER-ANALYTICS

---

## 📌 Business Problem

An online UK retailer needed answers to four key questions:
1. **Which customers are most valuable, and what is their lifetime value?**
2. **Can we predict future invoice sales from time and order features?**
3. **What does the next 3 months of revenue look like?**
4. **How can we proactively reduce customer churn?**

This project addresses all four using real transaction data from the [UCI Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/online+retail).

---

## 🎯 Project Highlights

- 📦 **Real-world dataset**: UCI Online Retail UK (541K rows → 397K after cleaning)
- 👥 **Customer Segmentation**: RFM Analysis + K-Means Clustering → High / Medium / Low Value segments
- 💰 **Customer Lifetime Value (CLV)**: Estimated CLV per customer using recency-weighted RFM formula
- 🤖 **Sales Prediction**: 4 regression models (incl. XGBoost) compared, selected by cross-validated R²
- 🧠 **SHAP Explainability**: Model predictions explained with SHapley Additive exPlanations
- 📅 **Time-Series Forecasting**: Prophet model generating 3-month revenue forecast with confidence intervals
- 🚨 **Churn Prediction**: Binary classifier on RFM features with ROC-AUC evaluation
- 🏭 **sklearn Pipeline**: Production-ready model artifact wrapping scaler + estimator in one object
- 📊 **Interactive Streamlit Dashboard**: 5 pages deployed on Streamlit Cloud

---

## 🏗️ Architecture

```
Raw Data (UCI Retail)
       │
       ▼
┌─────────────────┐     notebooks/01_cleaning.py
│  Data Cleaning  │ ──► data/cleaned_retail_sample.csv
└─────────────────┘
       │
       ├──► notebooks/02_eda.py          (EDA charts)
       │
       ├──► notebooks/03_segmentation.py (RFM + KMeans → data/rfm.csv)
       │
       ├──► notebooks/04_prediction.py   (XGBoost + SHAP + Pipeline → model.pkl)
       │
       ├──► notebooks/05_churn.py        (Churn classification)
       │
       ├──► notebooks/06_forecasting.py  (Prophet → data/forecast.csv)
       │
       └──► notebooks/07_clv.py          (CLV estimation → data/rfm.csv with CLV)
                 │
                 ▼
         app.py  (Streamlit — 5 pages)
                 │
                 ▼
         Streamlit Cloud (Live Demo)
```

---

## 🔬 Sales Prediction — ML Model Comparison

Models were evaluated on both held-out test R² and 5-fold cross-validated R² to detect overfitting.
**Best model selected by CV R²** (more robust than test-set R²).

| Model | Test R² | MAE | CV R² (5-fold) |
|---|---|---|---|
| Linear Regression | 0.40 | £17 | **0.52** ← best CV R², selected |
| Random Forest | 0.37 | £20 | 0.31 |
| Gradient Boosting | 0.25 | £20 | 0.15 |
| **XGBoost** | 0.27 | £19 | 0.17 |

> 💡 **Why CV R²?** Cross-validation gives a more honest estimate of generalization than a single test split. XGBoost is included for explainability via SHAP — tree-based boosting methods produce the most informative SHAP plots.

---

## 🧠 SHAP Model Explainability

SHAP (SHapley Additive exPlanations) provides feature-level attribution for every prediction:

- **Why did the model predict this invoice total?** → SHAP answers it
- Model-agnostic, theoretically grounded in game theory
- Used with XGBoost in this project (`notebooks/04_prediction.py`)
- Displayed as a bar summary plot in the Sales Prediction dashboard page

---

## 📅 Sales Forecasting — Prophet

- **Model**: Meta's Prophet with logistic growth + yearly seasonality
- **Horizon**: 3 months ahead with 90% confidence intervals
- **Output**: Interactive Plotly chart + forecast CSV (`data/forecast.csv`)
- **Script**: `notebooks/06_forecasting.py`

---

## 💰 Customer Lifetime Value (CLV)

CLV is estimated per customer using a recency-weighted formula:

> **CLV ≈ Monetary × Frequency × e^(−Recency / 365)**

- High-frequency, recently-active, high-spend customers score highest
- CLV distribution and segment breakdown shown in the Customer Segments page
- Script: `notebooks/07_clv.py`

---

## 🚨 Churn Prediction — Classification

- **Churn definition**: Customers with no purchase in the last **90 days** (Recency > 90)
- **Models trained**: Logistic Regression, Random Forest

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~88% | ~0.95 |
| Random Forest | ~89% | ~0.97 |

Key finding: **Recency is by far the most important churn predictor** (feature importance ~80%).

---

## 📊 Dashboard Pages

1. **Overview** — KPI metrics, interactive monthly revenue trend, top products, revenue by country
2. **Customer Segments** — RFM scatter, CLV metrics and distribution, segment breakdown, data table
3. **Sales Prediction** — Model comparison (4 models), SHAP explainability, live invoice predictor
4. **Sales Forecast** — Prophet 3-month revenue forecast with confidence intervals (interactive Plotly)
5. **Churn Analysis** — Confusion matrix, ROC curve, feature importance, live churn risk predictor

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn Pipeline, XGBoost, RandomForest, GradientBoosting, LogisticRegression |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Forecasting | Prophet (Meta) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
customer-analytics/
├── app.py                      # Streamlit dashboard (5 pages)
├── model.pkl                   # Best sales prediction pipeline (scaler + model)
├── requirements.txt
├── data/
│   ├── cleaned_retail_sample.csv
│   ├── rfm.csv                  # RFM + CLV data
│   ├── model_comparison.csv
│   ├── forecast.csv             # Prophet 3-month forecast output
│   └── high_risk_customers.csv
└── notebooks/
    ├── 01_cleaning.py           # Data cleaning pipeline
    ├── 02_eda.py                # Exploratory data analysis
    ├── 03_segmentation.py       # RFM + K-Means clustering
    ├── 04_prediction.py         # XGBoost + SHAP + sklearn Pipeline
    ├── 05_churn.py              # Churn classification models
    ├── 06_forecasting.py        # Prophet time-series forecasting
    ├── 07_clv.py                # Customer Lifetime Value estimation
    └── *.png                    # Generated charts (SHAP, forecast, CLV, etc.)
```

---

## 🔑 Key Learnings

- **RFM segmentation** is a simple but powerful framework for customer value analysis
- **XGBoost** consistently outperforms vanilla Random Forest on tabular data and is the most widely used algorithm in ML competitions and industry
- **SHAP** is the gold standard for ML explainability — it answers "why?" for any prediction, which is essential for stakeholder trust and model debugging
- **Prophet** handles real-world time-series challenges (seasonality, missing data, trend shifts) with minimal configuration — ideal for business forecasting
- **sklearn Pipelines** prevent data leakage, simplify deployment, and are the production-standard way to ship ML models
- **Cross-validation is essential** — test-set R² alone can be misleading; CV R² gives a more honest generalization estimate
- **Churn is highly predictable from recency alone** — a high recency value is a strong leading indicator

---

## 👤 Author

**Aman Pokhriyal**
- GitHub: https://github.com/amanwork5454-debug
