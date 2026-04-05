import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Load cleaned data (use sample CSV if full file not present)
_data_path = 'data/cleaned_retail.csv'
if not os.path.exists(_data_path):
    _data_path = 'data/cleaned_retail_sample.csv'

df = pd.read_csv(_data_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ── Aggregate to Monthly Revenue ──
df['MonthStart'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
monthly = (
    df.groupby('MonthStart')['TotalPrice']
    .sum()
    .reset_index()
    .rename(columns={'MonthStart': 'ds', 'TotalPrice': 'y'})
)
monthly = monthly.sort_values('ds').reset_index(drop=True)
print(f"Monthly data: {len(monthly)} periods")
print(monthly.tail())

# ── Train Prophet ──
# Add floor=0 so Prophet cannot forecast negative revenue
monthly['floor'] = 0
monthly['cap']   = monthly['y'].max() * 2.5   # generous headroom

m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    interval_width=0.90,
    growth='logistic'
)
m.fit(monthly)

# ── Forecast 3 Months Ahead ──
future = m.make_future_dataframe(periods=3, freq='MS')
future['floor'] = 0
future['cap']   = monthly['cap'].max()
forecast = m.predict(future)

# Mark which rows are actuals vs forecast
forecast['is_forecast'] = forecast['ds'] > monthly['ds'].max()

# ── Save Forecast Data for App ──
forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'is_forecast']].copy()
forecast_out['ds'] = forecast_out['ds'].dt.strftime('%Y-%m-%d')
forecast_out.to_csv('data/forecast.csv', index=False)
print("✅ Forecast data saved to data/forecast.csv")

# ── Save Forecast Plot ──
fig, ax = plt.subplots(figsize=(12, 5))

actuals = monthly.copy()
ax.plot(actuals['ds'], actuals['y'], 'o-', color='steelblue', label='Actual Revenue', linewidth=2)

hist_fc = forecast[~forecast['is_forecast']]
fut_fc  = forecast[forecast['is_forecast']]

ax.plot(hist_fc['ds'], hist_fc['yhat'], '--', color='orange', alpha=0.7, label='Fitted')
ax.fill_between(hist_fc['ds'], hist_fc['yhat_lower'], hist_fc['yhat_upper'],
                alpha=0.15, color='orange')

ax.plot(fut_fc['ds'], fut_fc['yhat'], 'o--', color='crimson', linewidth=2, label='3-Month Forecast')
ax.fill_between(fut_fc['ds'], fut_fc['yhat_lower'], fut_fc['yhat_upper'],
                alpha=0.25, color='crimson', label='90% CI')

ax.set_title('Monthly Revenue — Prophet Forecast (3-Month Horizon)', fontsize=13)
ax.set_xlabel('Month')
ax.set_ylabel('Revenue (£)')
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('notebooks/forecast.png', dpi=120, bbox_inches='tight')
plt.close()
print("✅ Forecast plot saved to notebooks/forecast.png")

# ── Print Forecast Summary ──
future_rows = forecast[forecast['is_forecast']][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
future_rows = future_rows.copy()
future_rows['ds'] = pd.to_datetime(future_rows['ds']).dt.strftime('%b %Y')
future_rows = future_rows.rename(columns={'ds': 'Month', 'yhat': 'Forecast_GBP',
                                          'yhat_lower': 'Lower_GBP', 'yhat_upper': 'Upper_GBP'})
future_rows = future_rows.round(0)
print("\n📊 3-Month Forecast:")
print(future_rows.to_string(index=False))
