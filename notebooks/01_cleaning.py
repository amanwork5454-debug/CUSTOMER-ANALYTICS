import pandas as pd

# Load dataset
df = pd.read_csv('data/online_retail.csv', encoding='latin-1')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove negative Quantity (returns/cancellations)
df = df[df['Quantity'] > 0]

# Remove negative UnitPrice
df = df[df['UnitPrice'] > 0]

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("\nAfter cleaning:")
print("Shape:", df.shape)
print(df.head())

# Save cleaned data
df.to_csv('data/cleaned_retail.csv', index=False)
print("\n✅ Cleaned data saved to data/cleaned_retail.csv")