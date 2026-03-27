import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
st.title("🛍️ Customer Analytics & Sales Prediction")

# ── Load Data ──
df = pd.read_csv('data/cleaned_retail_sample.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
rfm = pd.read_csv('data/rfm.csv')

# ── Load Model ──
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
model_name = model_data['model_name']
model_r2 = model_data['r2']
model_mae = model_data['mae']
model_rmse = model_data['rmse']

# ── Sidebar ──
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Customer Segments", "Sales Prediction"])

# ══════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════
if page == "Overview":
    st.header("📊 Business Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")
    col2.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
    col3.metric("Total Orders", f"{df['InvoiceNo'].nunique():,}")

    st.subheader("Monthly Sales Trend")
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly = df.groupby('MonthYear')['TotalPrice'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=monthly, x='MonthYear', y='TotalPrice', marker='o', ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Top 10 Products")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_products.values, y=top_products.index, ax=ax2, palette='viridis', hue=top_products.index, legend=False)
    plt.tight_layout()
    st.pyplot(fig2)

# ══════════════════════════════
# PAGE 2: CUSTOMER SEGMENTS
# ══════════════════════════════
elif page == "Customer Segments":
    st.header("👥 Customer Segmentation (RFM)")
    col1, col2, col3 = st.columns(3)
    col1.metric("High Value Customers", len(rfm[rfm['Segment'] == 'High Value']))
    col2.metric("Medium Value Customers", len(rfm[rfm['Segment'] == 'Medium Value']))
    col3.metric("Low Value Customers", len(rfm[rfm['Segment'] == 'Low Value']))

    st.subheader("Customer Segments Scatter Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='Set1', ax=ax)
    st.pyplot(fig)

    st.subheader("RFM Data Table")
    st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']].head(50))

# ══════════════════════════════
# PAGE 3: SALES PREDICTION
# ══════════════════════════════
elif page == "Sales Prediction":
    st.header("📈 Sales Prediction")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", model_name)
    col2.metric("R² Score", f"{model_r2:.2f}")
    col3.metric("MAE", f"£{float(model_mae):,.0f}")
    

    st.subheader("Predict Invoice Sales")
    c1, c2, c3, c4 = st.columns(4)
    year         = c1.selectbox("Year", [2011, 2012])
    month        = c2.selectbox("Month", list(range(1, 13)))
    num_items    = c3.number_input("Num Items", min_value=1, value=20)
    num_products = c4.number_input("Num Products", min_value=1, value=5)

    day_of_week  = st.selectbox("Day of Week", [0,1,2,3,4,5,6],
                    format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    day_of_month = st.slider("Day of Month", 1, 31, 15)
    quarter      = (month - 1) // 3 + 1
    is_weekend   = 1 if day_of_week >= 5 else 0

    if st.button("Predict Sales"):
        features = [[year, month, day_of_week, day_of_month,
                     quarter, is_weekend, num_items, num_products]]
        pred = model.predict(features)[0]
        st.success(f"Predicted Invoice Sales: £{pred:,.2f}")

    st.subheader("Actual vs Predicted (Test Set)")
    st.image('notebooks/prediction.png')