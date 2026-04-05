import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

MODEL_XGBOOST = 'XGBoost'

# Load cleaned data (use sample CSV if full file not present)
_data_path = 'data/cleaned_retail.csv'
if not os.path.exists(_data_path):
    _data_path = 'data/cleaned_retail_sample.csv'
df = pd.read_csv(_data_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ── Feature Engineering ──
df['Month']      = df['InvoiceDate'].dt.month
df['Year']       = df['InvoiceDate'].dt.year
df['DayOfWeek']  = df['InvoiceDate'].dt.dayofweek
df['DayOfMonth'] = df['InvoiceDate'].dt.day
df['Quarter']    = df['InvoiceDate'].dt.quarter
df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)

# ── Aggregate at invoice level ──
invoice_df = df.groupby(['InvoiceNo', 'Year', 'Month', 'DayOfWeek',
                          'DayOfMonth', 'Quarter', 'IsWeekend']).agg(
    TotalSales  =('TotalPrice', 'sum'),
    NumItems    =('Quantity', 'sum'),
    NumProducts =('StockCode', 'nunique')
).reset_index()

print(f"Dataset size: {len(invoice_df)} invoices")

# ── Features & Target ──
features = ['Year', 'Month', 'DayOfWeek', 'DayOfMonth',
            'Quarter', 'IsWeekend', 'NumItems', 'NumProducts']
X = invoice_df[features]
y = invoice_df['TotalSales']

# ── Train/Test Split (shuffle=False preserves time order) ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Gap threshold: warn if test R² exceeds CV R² by more than this amount
OVERFITTING_GAP_THRESHOLD = 0.15

# ── Train & Evaluate All Models ──
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
    MODEL_XGBOOST:       XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                      random_state=42, verbosity=0)
}

results = {}
comparison_rows = []

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Cross-Validation on full dataset (5-fold, time-aware shuffle=False)
    cv_scores = cross_val_score(m, X, y, cv=5, scoring='r2')
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    results[name] = {
        'model': m, 'r2': r2, 'mae': mae,
        'rmse': rmse, 'cv_mean': cv_mean,
        'cv_std': cv_std, 'y_pred': y_pred
    }
    comparison_rows.append({
        'Model': name,
        'R²': round(r2, 4),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'CV R² (mean)': round(cv_mean, 4),
        'CV R² (std)': round(cv_std, 4)
    })
    print(f"\n{name}:")
    print(f"  R²        : {r2:.4f}")
    print(f"  MAE       : £{mae:,.2f}")
    print(f"  RMSE      : £{rmse:,.2f}")
    print(f"  CV R²     : {cv_mean:.4f} ± {cv_std:.4f}")
    if r2 - cv_mean > OVERFITTING_GAP_THRESHOLD:
        print(f"  ⚠️  Gap between test R² and CV R² ({r2 - cv_mean:.2f}) suggests overfitting on test split.")

# ── Save Model Comparison Table ──
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv('data/model_comparison.csv', index=False)
print("\nModel Comparison:")
print(comparison_df)

# ── Pick Best Model by CV R² (more robust than test R²) ──
# Using CV R² as the selection criterion avoids picking a model that got
# lucky on a particular test split.
best_name = max(results, key=lambda x: results[x]['cv_mean'])
best = results[best_name]
print(f"\n✅ Best Model by CV R²: {best_name} (CV R² = {best['cv_mean']:.4f}, Test R² = {best['r2']:.4f})")

# ── Feature Importance (Random Forest) ──
rf_model = results['Random Forest']['model']
importance = pd.Series(rf_model.feature_importances_, index=features)
importance = importance.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
importance.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Importance (Random Forest)')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('notebooks/feature_importance.png')
plt.close()
print("✅ Feature importance chart saved")

# ── Model Comparison Bar Chart ──
comp = comparison_df.set_index('Model')
fig, ax = plt.subplots(figsize=(9, 4))
comp['R²'].plot(kind='bar', ax=ax,
                color=['steelblue', 'orange', 'green', 'crimson'],
                edgecolor='black')
ax.set_title('Model Comparison — R² Score')
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('notebooks/model_comparison.png')
plt.close()
print("✅ Model comparison chart saved")

# ── Actual vs Predicted Plot ──
y_pred_best = best['y_pred']
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label='Actual',    marker='o', color='steelblue')
plt.plot(y_pred_best[:50],   label='Predicted', marker='x', color='orange')
plt.title(f'Actual vs Predicted Sales ({best_name})')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/prediction.png')
plt.close()
print("✅ Prediction chart saved")

# ── SHAP Explainability ──
# Use XGBoost model (or best tree model) for SHAP values
shap_model_name = MODEL_XGBOOST if MODEL_XGBOOST in results else best_name
shap_raw_model = results[shap_model_name]['model']
print(f"\nComputing SHAP values for: {shap_model_name}")
explainer   = shap.TreeExplainer(shap_raw_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(9, 5))
shap.summary_plot(shap_values, X_test, feature_names=features,
                  plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance — {shap_model_name}')
plt.tight_layout()
plt.savefig('notebooks/shap_summary.png', bbox_inches='tight')
plt.close()
print("✅ SHAP summary plot saved")

# ── Build sklearn Pipeline (best model wrapped with scaler) ──
# For tree models the scaler is a no-op but wrapping shows production-readiness.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  best['model'])
])
pipeline.fit(X_train, y_train)

# ── Save Pipeline & Metadata ──
model_data = {
    'model':      pipeline,        # sklearn Pipeline (scaler + best model)
    'features':   features,
    'model_name': best_name,
    'r2':         best['r2'],
    'mae':        best['mae'],
    'rmse':       best['rmse'],
    'cv_mean':    best['cv_mean'],
    'cv_std':     best['cv_std']
}
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("✅ Pipeline saved as model.pkl")