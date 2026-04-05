import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data (fall back to sample CSV if full file is not present)
_data_path = 'data/cleaned_retail.csv'
if not os.path.exists(_data_path):
    _data_path = 'data/cleaned_retail_sample.csv'
df = pd.read_csv(_data_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ── RFM Calculation ──
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo':   'nunique',                                  # Frequency
    'TotalPrice':  'sum'                                       # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print("RFM Table:")
print(rfm.head())

# ── Scaling ──
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# ── Elbow Method ──
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o', color='steelblue')
plt.title('Elbow Method - Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig('notebooks/elbow.png')
plt.show()
print("✅ Elbow chart saved")

# ── K-Means with 3 clusters ──
km = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = km.fit_predict(rfm_scaled)

# Label clusters
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("\nCluster Summary:")
print(cluster_summary)

rfm['Segment'] = rfm['Cluster'].map({
    rfm.groupby('Cluster')['Monetary'].mean().idxmax(): 'High Value',
    rfm.groupby('Cluster')['Monetary'].mean().idxmin(): 'Low Value',
})
rfm['Segment'] = rfm['Segment'].fillna('Medium Value')

# ── Plot Segments ──
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='Set1')
plt.title('Customer Segments')
plt.tight_layout()
plt.savefig('notebooks/segments.png')
plt.show()
print("✅ Segmentation chart saved")

# Save RFM data
rfm.to_csv('data/rfm.csv', index=False)
print("✅ RFM data saved to data/rfm.csv")