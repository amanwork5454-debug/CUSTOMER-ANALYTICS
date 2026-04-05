import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data (fall back to sample CSV if full file is not present)
_data_path = 'data/cleaned_retail.csv'
if not os.path.exists(_data_path):
    _data_path = 'data/cleaned_retail_sample.csv'
df = pd.read_csv(_data_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract month-year
df['MonthYear'] = df['InvoiceDate'].dt.to_period('M')

# ── 1. Monthly Sales Trend ──
monthly_sales = df.groupby('MonthYear')['TotalPrice'].sum().reset_index()
monthly_sales['MonthYear'] = monthly_sales['MonthYear'].astype(str)

plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly_sales, x='MonthYear', y='TotalPrice', marker='o', color='steelblue')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('notebooks/monthly_sales.png')
plt.show()
print("✅ Monthly sales chart saved")

# ── 2. Top 10 Products ──
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 10 Best-Selling Products')
plt.xlabel('Total Quantity Sold')
plt.tight_layout()
plt.savefig('notebooks/top_products.png')
plt.show()
print("✅ Top products chart saved")

# ── 3. Top 10 Customers ──
top_customers = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_customers.values, y=top_customers.index.astype(str), palette='magma')
plt.title('Top 10 Customers by Revenue')
plt.xlabel('Total Spend')
plt.tight_layout()
plt.savefig('notebooks/top_customers.png')
plt.show()
print("✅ Top customers chart saved")