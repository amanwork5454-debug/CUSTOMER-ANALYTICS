import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Load cleaned data
df = pd.read_csv('data/cleaned_retail.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ── Feature Engineering (row-level, not monthly) ──
df['Month']       = df['InvoiceDate'].dt.month
df['Year']        = df['InvoiceDate'].dt.year
df['DayOfWeek']   = df['InvoiceDate'].dt.dayofweek
df['DayOfMonth']  = df['InvoiceDate'].dt.day
df['Quarter']     = df['InvoiceDate'].dt.quarter
df['IsWeekend']   = (df['DayOfWeek'] >= 5).astype(int)

# ── Aggregate at invoice level ──
invoice_df = df.groupby(['InvoiceNo', 'Year', 'Month', 'DayOfWeek',
                          'DayOfMonth', 'Quarter', 'IsWeekend']).agg(
    TotalSales=('TotalPrice', 'sum'),
    NumItems=('Quantity', 'sum'),
    NumProducts=('StockCode', 'nunique')
).reset_index()

print(f"Dataset size: {len(invoice_df)} invoices")
print(invoice_df.head())

# ── Features & Target ──
features = ['Year', 'Month', 'DayOfWeek', 'DayOfMonth',
            'Quarter', 'IsWeekend', 'NumItems', 'NumProducts']
X = invoice_df[features]
y = invoice_df['TotalSales']

# ── Train/Test Split (80/20) ──
split = int(0.8 * len(invoice_df))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── Train Multiple Models ──
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'model': m, 'r2': r2, 'mae': mae, 'rmse': rmse, 'y_pred': y_pred}
    print(f"\n{name}:")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : £{mae:,.2f}")
    print(f"  RMSE : £{rmse:,.2f}")

# ── Pick Best Model ──
best_name = max(results, key=lambda x: results[x]['r2'])
best = results[best_name]
print(f"\n✅ Best Model: {best_name} (R² = {best['r2']:.4f})")

# ── Feature Importance ──
if hasattr(best['model'], 'feature_importances_'):
    importance = pd.Series(best['model'].feature_importances_, index=features)
    importance = importance.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    plt.savefig('notebooks/feature_importance.png')
    plt.show()
    print("✅ Feature importance chart saved")

# ── Actual vs Predicted Plot ──
y_pred_best = best['y_pred']
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label='Actual', marker='o', color='steelblue')
plt.plot(y_pred_best[:50],   label='Predicted', marker='x', color='orange')
plt.title(f'Actual vs Predicted Sales ({best_name})')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/prediction.png')
plt.show()
print("✅ Prediction chart saved")

# ── Save Best Model + Metadata ──
model_data = {
    'model': best['model'],
    'features': features,
    'model_name': best_name,
    'r2': best['r2'],
    'mae': best['mae'],
    'rmse': best['rmse']
}
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("✅ Model saved as model.pkl")