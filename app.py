import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)

st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
st.title("🛍️ Customer Analytics & Sales Prediction")

# ── Load Data ──
df = pd.read_csv('data/cleaned_retail_sample.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
rfm = pd.read_csv('data/rfm.csv')
comparison = pd.read_csv('data/model_comparison.csv')

# ── Load Model (sklearn Pipeline) ──
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model      = model_data['model']   # sklearn Pipeline (scaler + best model)
model_name = model_data['model_name']
model_r2   = float(model_data['r2'])
model_mae  = float(model_data['mae'])
model_rmse = float(model_data['rmse'])
cv_mean    = float(model_data['cv_mean'])
cv_std     = float(model_data['cv_std'])

@st.cache_resource
def train_churn_models(_rfm):
    """Train churn classifiers once and cache across page loads."""
    churn_df = _rfm[['Recency', 'Frequency', 'Monetary']].copy()
    churn_df['Churned'] = (_rfm['Recency'] > 90).astype(int)
    X_c = churn_df[['Recency', 'Frequency', 'Monetary']]
    y_c = churn_df['Churned']
    scaler_c = StandardScaler()
    X_scaled = scaler_c.fit_transform(X_c)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_scaled, y_c, test_size=0.2, random_state=42, stratify=y_c
    )
    lr_clf = LogisticRegression(random_state=42, max_iter=1000)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_clf.fit(X_train_c, y_train_c)
    rf_clf.fit(X_train_c, y_train_c)
    return scaler_c, X_scaled, X_train_c, X_test_c, y_train_c, y_test_c, lr_clf, rf_clf

# ── Sidebar ──
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Customer Segments", "Sales Prediction",
                                   "Sales Forecast", "Churn Analysis"])

# ══════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════
if page == "Overview":
    st.header("📊 Business Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")
    col2.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
    col3.metric("Total Orders", f"{df['InvoiceNo'].nunique():,}")
    col4.metric("Avg Order Value", f"£{df.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.0f}")

    st.subheader("Monthly Sales Trend")
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly = df.groupby('MonthYear')['TotalPrice'].sum().reset_index()
    fig_line = px.line(
        monthly, x='MonthYear', y='TotalPrice',
        markers=True, labels={'TotalPrice': 'Revenue (£)', 'MonthYear': 'Month'},
        title='Monthly Revenue Trend'
    )
    fig_line.update_traces(line_color='steelblue', marker_color='steelblue')
    fig_line.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_line, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 10 Products by Quantity")
        top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_prod = px.bar(
            top_products, x='Quantity', y='Description', orientation='h',
            color='Quantity', color_continuous_scale='viridis',
            labels={'Description': 'Product', 'Quantity': 'Units Sold'}
        )
        fig_prod.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_b:
        st.subheader("Revenue by Country (Top 10)")
        top_countries = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_country = px.bar(
            top_countries, x='TotalPrice', y='Country', orientation='h',
            color='TotalPrice', color_continuous_scale='Blues',
            labels={'TotalPrice': 'Revenue (£)', 'Country': 'Country'}
        )
        fig_country.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        st.plotly_chart(fig_country, use_container_width=True)

# ══════════════════════════════
# PAGE 2: CUSTOMER SEGMENTS
# ══════════════════════════════
elif page == "Customer Segments":
    st.header("👥 Customer Segmentation (RFM)")
    col1, col2, col3 = st.columns(3)
    col1.metric("High Value Customers",   len(rfm[rfm['Segment'] == 'High Value']))
    col2.metric("Medium Value Customers", len(rfm[rfm['Segment'] == 'Medium Value']))
    col3.metric("Low Value Customers",    len(rfm[rfm['Segment'] == 'Low Value']))

    # ── CLV Metric Cards ──
    if 'CLV_Estimated' in rfm.columns:
        st.subheader("💰 Customer Lifetime Value (CLV)")
        st.markdown(
            "Estimated CLV uses the formula: **CLV ≈ Monetary × Frequency × e^(−Recency/365)** "
            "— rewarding frequent, high-spend, recently active customers."
        )
        avg_clv_col, median_clv_col, high_value_avg_col, total_portfolio_col = st.columns(4)
        avg_clv_col.metric("Avg CLV",    f"£{rfm['CLV_Estimated'].mean():,.0f}")
        median_clv_col.metric("Median CLV", f"£{rfm['CLV_Estimated'].median():,.0f}")
        high_value_avg_col.metric("High Value Avg CLV",
                        f"£{rfm[rfm['Segment']=='High Value']['CLV_Estimated'].mean():,.0f}")
        total_portfolio_col.metric("Total Portfolio CLV",
                        f"£{rfm['CLV_Estimated'].sum():,.0f}")

        st.subheader("CLV Distribution by Segment")
        st.image('notebooks/clv_distribution.png', use_container_width=True)

    st.subheader("Customer Segments Scatter Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary',
                    hue='Segment', palette='Set1', ax=ax)
    st.pyplot(fig)

    st.subheader("RFM Data Table")
    st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency',
                       'Monetary', 'Segment']].head(50))

# ══════════════════════════════
# PAGE 3: SALES PREDICTION
# ══════════════════════════════
elif page == "Sales Prediction":
    st.header("📈 Sales Prediction")

    # ── Model Performance Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", model_name)
    col2.metric("R² Score", f"{model_r2:.2f}")
    col3.metric("MAE", f"£{model_mae:,.0f}")
    col4.metric("CV R² (5-fold)", f"{cv_mean:.2f} ± {cv_std:.2f}")

    # ── Model Comparison Table ──
    st.subheader("📊 Model Comparison")
    st.dataframe(comparison, use_container_width=True)

    # ── Model Comparison Chart ──
    st.image('notebooks/model_comparison.png')

    # ── Feature Importance ──
    st.subheader("🔍 Feature Importance (Random Forest)")
    st.image('notebooks/feature_importance.png')

    # ── Predict ──
    st.subheader("🔮 Predict Invoice Sales")
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
        st.success(f"🎯 Predicted Invoice Sales: £{pred:,.2f}")

    st.subheader("Actual vs Predicted (Test Set)")
    st.image('notebooks/prediction.png')

    # ── Model Explainability (SHAP) ──
    st.subheader("🧠 Model Explainability — SHAP Values (XGBoost)")
    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** shows the contribution of each feature "
        "to the model's prediction. Longer bars = more influential features. "
        "Unlike standard feature importance, SHAP is model-agnostic and theoretically grounded."
    )
    st.image('notebooks/shap_summary.png', use_container_width=True)
    st.info(
        "💡 **Interview tip**: SHAP is the industry standard for explaining ML predictions. "
        "It answers 'why did the model predict X?' — critical for stakeholder communication "
        "and model debugging."
    )

# ══════════════════════════════
# PAGE 4: SALES FORECAST
# ══════════════════════════════
elif page == "Sales Forecast":
    st.header("📅 Time-Series Sales Forecast (Prophet)")
    st.markdown(
        "Using **Meta's Prophet** library, we fit a logistic-growth model on monthly revenue "
        "and generate a **3-month forward forecast** with 90% confidence intervals."
    )

    try:
        forecast_df = pd.read_csv('data/forecast.csv')
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

        actuals = forecast_df[~forecast_df['is_forecast']].copy()
        future  = forecast_df[forecast_df['is_forecast']].copy()

        # ── Monthly Revenue KPIs ──
        last_actual_rev = actuals['yhat'].iloc[-1]
        next_month_rev  = future['yhat'].iloc[0] if len(future) > 0 else 0
        total_forecast  = future['yhat'].sum()

        last_month_col, next_month_col, total_forecast_col = st.columns(3)
        last_month_col.metric("Last Actual Month (£)", f"£{last_actual_rev:,.0f}")
        next_month_col.metric("Next Month Forecast",   f"£{next_month_rev:,.0f}")
        total_forecast_col.metric("3-Month Total Forecast", f"£{total_forecast:,.0f}")

        # ── Interactive Plotly Forecast Chart ──
        st.subheader("Interactive Forecast Chart")
        # Load actual monthly revenue from the transaction data
        df['MonthStart'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
        monthly_actual = df.groupby('MonthStart')['TotalPrice'].sum().reset_index()
        monthly_actual.columns = ['ds', 'actual_revenue']

        fig_fc = go.Figure()

        # Actual revenue line
        fig_fc.add_trace(go.Scatter(
            x=monthly_actual['ds'], y=monthly_actual['actual_revenue'],
            name='Actual Revenue', mode='lines+markers',
            line=dict(color='steelblue', width=2), marker=dict(size=6)
        ))

        # Fitted (historical) with CI
        fig_fc.add_trace(go.Scatter(
            x=actuals['ds'], y=actuals['yhat_upper'],
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_fc.add_trace(go.Scatter(
            x=actuals['ds'], y=actuals['yhat_lower'],
            fill='tonexty', fillcolor='rgba(255,165,0,0.15)',
            line=dict(width=0), name='Historical CI', hoverinfo='skip'
        ))

        # Forecast with CI
        fig_fc.add_trace(go.Scatter(
            x=future['ds'], y=future['yhat_upper'],
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_fc.add_trace(go.Scatter(
            x=future['ds'], y=future['yhat_lower'],
            fill='tonexty', fillcolor='rgba(220,20,60,0.2)',
            line=dict(width=0), name='Forecast 90% CI', hoverinfo='skip'
        ))
        fig_fc.add_trace(go.Scatter(
            x=future['ds'], y=future['yhat'],
            name='3-Month Forecast', mode='lines+markers',
            line=dict(color='crimson', width=2, dash='dot'),
            marker=dict(size=8, symbol='diamond')
        ))

        fig_fc.update_layout(
            xaxis_title='Month', yaxis_title='Revenue (£)',
            title='Monthly Revenue — Prophet Forecast',
            legend=dict(x=0.01, y=0.99), height=420
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Forecast Table ──
        st.subheader("📋 3-Month Forecast Detail")
        display_future = future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        display_future['ds'] = display_future['ds'].dt.strftime('%B %Y')
        display_future = display_future.rename(columns={
            'ds': 'Month', 'yhat': 'Forecast (£)',
            'yhat_lower': 'Lower Bound (£)', 'yhat_upper': 'Upper Bound (£)'
        })
        display_future[['Forecast (£)', 'Lower Bound (£)', 'Upper Bound (£)']] = \
            display_future[['Forecast (£)', 'Lower Bound (£)', 'Upper Bound (£)']].round(0)
        st.dataframe(display_future, use_container_width=True)

        st.info(
            "💡 **About Prophet**: Developed by Meta, Prophet handles seasonality, holidays, "
            "and trend changes out-of-the-box. It's widely used in industry for business "
            "forecasting tasks. Run `notebooks/06_forecasting.py` to regenerate."
        )

    except FileNotFoundError:
        st.warning("Forecast data not found. Run `notebooks/06_forecasting.py` to generate it.")

# ══════════════════════════════
# PAGE 5: CHURN ANALYSIS
# ══════════════════════════════
elif page == "Churn Analysis":
    st.header("🚨 Customer Churn Analysis")
    st.markdown(
        "Customers who haven't purchased in **90+ days** are classified as **churned**. "
        "We train a classifier on RFM features to predict churn risk."
    )
    st.info(
        "📌 **Note on model performance**: Since the churn label is derived directly from Recency "
        "(Recency > 90 days), the classifier achieves near-perfect accuracy — it essentially "
        "re-learns this threshold. In a real production system, you would incorporate additional "
        "features (product categories, return rates, browsing behaviour) to make the problem "
        "genuinely predictive. This page demonstrates the full classification pipeline."
    )

    # ── Prepare churn data ──
    churn_df = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    churn_df['Churned'] = (rfm['Recency'] > 90).astype(int)

    total = len(churn_df)
    churned_count = churn_df['Churned'].sum()
    active_count = total - churned_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned (>90 days)", f"{churned_count:,}", delta=f"{churned_count/total*100:.1f}%", delta_color="inverse")
    col3.metric("Active Customers", f"{active_count:,}", delta=f"{active_count/total*100:.1f}%")

    # ── Load cached classifiers (trained once per session) ──
    scaler_c, X_scaled, X_train_c, X_test_c, y_train_c, y_test_c, lr_clf, rf_clf = train_churn_models(rfm)

    lr_pred = lr_clf.predict(X_test_c)
    rf_pred = rf_clf.predict(X_test_c)
    lr_prob = lr_clf.predict_proba(X_test_c)[:, 1]
    rf_prob = rf_clf.predict_proba(X_test_c)[:, 1]

    lr_auc = roc_auc_score(y_test_c, lr_prob)
    rf_auc = roc_auc_score(y_test_c, rf_prob)

    # ── Model Metrics ──
    st.subheader("📊 Classifier Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LR Accuracy",  f"{accuracy_score(y_test_c, lr_pred):.2%}")
    m2.metric("LR ROC-AUC",   f"{lr_auc:.3f}")
    m3.metric("RF Accuracy",  f"{accuracy_score(y_test_c, rf_pred):.2%}")
    m4.metric("RF ROC-AUC",   f"{rf_auc:.3f}")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.subheader("Confusion Matrix (Random Forest)")
        cm = confusion_matrix(y_test_c, rf_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Active', 'Churned'],
                    yticklabels=['Active', 'Churned'])
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig_cm)

    with col_roc:
        st.subheader("ROC Curve")
        fpr_lr, tpr_lr, _ = roc_curve(y_test_c, lr_prob)
        fpr_rf, tpr_rf, _ = roc_curve(y_test_c, rf_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f'Logistic Regression (AUC={lr_auc:.3f})', line=dict(color='steelblue')))
        fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name=f'Random Forest (AUC={rf_auc:.3f})', line=dict(color='orange')))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(dash='dash', color='grey')))
        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                              legend=dict(x=0.4, y=0.1), height=350)
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Feature Importance (RF) ──
    st.subheader("🔍 Feature Importance for Churn")
    feat_imp = pd.Series(rf_clf.feature_importances_,
                         index=['Recency', 'Frequency', 'Monetary']).sort_values(ascending=True)
    fig_imp = px.bar(feat_imp, orientation='h',
                     labels={'value': 'Importance', 'index': 'Feature'},
                     color=feat_imp.values, color_continuous_scale='Blues')
    fig_imp.update_layout(coloraxis_showscale=False, showlegend=False, height=250)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Churn Distribution ──
    st.subheader("📌 Churn Classification Report")
    report = classification_report(y_test_c, rf_pred,
                                   target_names=['Active', 'Churned'], output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision','recall','f1-score']), use_container_width=True)

    # ── Interactive Churn Predictor ──
    st.subheader("🔮 Predict Churn Risk for a Customer")
    st.markdown("Enter a customer's RFM values to get their churn probability:")
    p1, p2, p3 = st.columns(3)
    inp_recency   = p1.number_input("Recency (days since last purchase)", min_value=1, max_value=400, value=50)
    inp_frequency = p2.number_input("Frequency (number of orders)",       min_value=1, max_value=200, value=5)
    inp_monetary  = p3.number_input("Monetary (total spend £)",           min_value=1.0, max_value=100000.0, value=500.0)

    if st.button("Predict Churn Risk"):
        inp_scaled = scaler_c.transform([[inp_recency, inp_frequency, inp_monetary]])
        churn_prob = rf_clf.predict_proba(inp_scaled)[0][1]
        if churn_prob >= 0.6:
            st.error(f"⚠️ High Churn Risk — {churn_prob:.1%} probability of churning. Consider a re-engagement campaign.")
        elif churn_prob >= 0.35:
            st.warning(f"🟡 Medium Churn Risk — {churn_prob:.1%} probability. Monitor this customer.")
        else:
            st.success(f"✅ Low Churn Risk — {churn_prob:.1%} probability. Customer looks healthy!")