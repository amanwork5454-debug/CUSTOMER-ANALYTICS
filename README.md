# 🛍️ Customer Analytics & Sales Prediction

An end-to-end data science project analyzing 500K+ retail transactions to uncover customer behavior patterns and predict future sales.

🔗 **Live Demo:** https://aman-customer-analytics.streamlit.app
📂 **GitHub:** https://github.com/amanwork5454-debug/CUSTOMER-ANALYTICS

---

## 📌 Project Overview
This project analyzes the Online Retail UK dataset (541K rows, cleaned to 397K) to deliver actionable business insights through customer segmentation and sales forecasting.

---

## ✨ Features
- 🧹 **Data Cleaning** — Handled missing values, duplicates, outliers
- 📈 **Exploratory Data Analysis** — Monthly sales trends, top products, top customers
- 👥 **Customer Segmentation** — RFM Analysis + K-Means Clustering (3 segments)
- 🤖 **Sales Prediction** — Random Forest Regressor (R² = 0.30)
- 🖥️ **Interactive Dashboard** — 3-page Streamlit app deployed on cloud

---

## 🛠️ Tech Stack
| Category | Tools |
|----------|-------|
| Language | Python |
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |

---

## 📊 Dataset
- **Source:** UCI Online Retail UK Dataset
- **Size:** 541K rows → 397K after cleaning
- **Period:** Dec 2010 – Dec 2011

---

## 📁 Project Structure
```
customer-analytics/
├── app.py                  # Streamlit dashboard
├── model.pkl               # Trained ML model
├── requirements.txt
├── data/
│   ├── cleaned_retail_sample.csv
│   └── rfm.csv
└── notebooks/
    ├── 01_cleaning.py
    ├── 02_eda.py
    ├── 03_segmentation.py
    └── 04_prediction.py
```

---

## 👤 Author
**Aman Pokhriyal**
📧 amanwork5454@gmail.com