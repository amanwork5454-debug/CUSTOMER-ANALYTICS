# 🛍️ Customer Analytics & Sales Prediction

An end-to-end Data Science project analyzing 500K+ retail transactions to uncover customer behavior and predict sales using Machine Learning.

🔗 **Live Demo:** https://aman-customer-analytics.streamlit.app
📂 **GitHub:** https://github.com/amanwork5454-debug/CUSTOMER-ANALYTICS

---

## 📌 Project Highlights
- 📦 Real-world dataset: UCI Online Retail UK (541K rows → 397K after cleaning)
- 👥 Customer Segmentation using RFM Analysis + K-Means Clustering
- 🤖 Sales Prediction with 3 ML models compared
- 📊 Interactive Streamlit dashboard deployed on cloud
- ✅ Best Model: Linear Regression — R² = 0.83, CV R² = 0.65

---

## 🔬 ML Model Comparison

| Model | R² Score | MAE | CV R² |
|-------|----------|-----|-------|
| Linear Regression | **0.83** | £187 | 0.65 |
| Gradient Boosting | 0.68 | £185 | 0.45 |
| Random Forest | 0.56 | £181 | 0.60 |

---

## 🧠 Features Used for Prediction
- Year, Month, Quarter
- Day of Week, Day of Month, Is Weekend
- Number of Items, Number of Products

---

## 📊 Dashboard Pages
1. **Overview** — Revenue, customers, monthly sales trend, top products
2. **Customer Segments** — RFM scatter plot, segment breakdown, data table
3. **Sales Prediction** — Model metrics, feature importance, model comparison, live prediction

---

## 🛠️ Tech Stack
- **Data:** Python, Pandas, NumPy
- **ML:** Scikit-learn (LinearRegression, RandomForest, GradientBoosting)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit Cloud
- **Version Control:** Git, GitHub

---

## 📁 Project Structure
```
customer-analytics/
├── app.py                    # Streamlit dashboard
├── model.pkl                 # Trained ML model + metrics
├── requirements.txt
├── data/
│   ├── cleaned_retail_sample.csv
│   ├── rfm.csv
│   └── model_comparison.csv
└── notebooks/
    ├── 01_cleaning.py
    ├── 02_eda.py
    ├── 03_segmentation.py
    ├── 04_prediction.py
    ├── feature_importance.png
    ├── model_comparison.png
    └── prediction.png
```

## 👤 Author
**Aman Pokhriyal**
- GitHub: https://github.com/amanwork5454-debug