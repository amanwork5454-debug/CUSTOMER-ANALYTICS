import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

# Load cleaned data
df = pd.read_csv('data/cleaned_retail.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ── Create monthly sales data ──
df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year

monthly = df.groupby(['Year', 'Month'])['TotalPrice'].sum().reset_index()
monthly.columns = ['Year', 'Month', 'TotalSales']

print("Monthly Sales Data:")
print(monthly)

# ── Features & Target ──
X = monthly[['Year', 'Month']]
y = monthly['TotalSales']

# ── Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train Model ──
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ──
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"MAE  : £{mae:,.2f}")
print(f"R²   : {r2:.4f}")

# ── Plot Actual vs Predicted ──
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', marker='o', color='steelblue')
plt.plot(y_pred, label='Predicted', marker='x', color='orange')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/prediction.png')
plt.show()
print("✅ Prediction chart saved")

# ── Save Model ──
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved as model.pkl")