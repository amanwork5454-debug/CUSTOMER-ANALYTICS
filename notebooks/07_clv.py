import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load RFM Data ──
rfm = pd.read_csv('data/rfm.csv')
print(f"RFM records: {len(rfm)}")
print(rfm.head())

# ── Simplified CLV Estimation ──
# CLV proxy = Monetary × (Frequency / max_recency_penalty)
# We use a regression approach: predict future Monetary from R, F, M history.
#
# Formula intuition:
#   - High Frequency → customer buys often → higher CLV
#   - Low Recency    → customer bought recently → higher CLV multiplier
#   - High Monetary  → high past spend → likely high future spend
#
# Recency penalty: recency_weight = exp(-Recency / 365)
# Predicted annual CLV ≈ Monetary × Frequency × recency_weight

rfm['RecencyWeight'] = np.exp(-rfm['Recency'] / 365.0)
rfm['CLV_Estimated'] = (rfm['Monetary'] * rfm['Frequency'] * rfm['RecencyWeight']).round(2)

# Cap extreme outliers at 99th percentile for display
p99 = rfm['CLV_Estimated'].quantile(0.99)
rfm['CLV_Display'] = rfm['CLV_Estimated'].clip(upper=p99)

print(f"\nCLV Stats:")
print(rfm['CLV_Estimated'].describe().round(2))

# ── CLV by Segment ──
clv_by_segment = rfm.groupby('Segment')['CLV_Estimated'].agg(['mean', 'median', 'sum']).round(2)
clv_by_segment.columns = ['Mean CLV', 'Median CLV', 'Total CLV']
print("\nCLV by Segment:")
print(clv_by_segment)

# ── CLV Distribution Plot ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(rfm['CLV_Display'], bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('CLV Distribution (capped at 99th pct)')
axes[0].set_xlabel('Estimated CLV (£)')
axes[0].set_ylabel('Number of Customers')

# CLV by Segment boxplot
sns.boxplot(data=rfm, x='Segment', y='CLV_Display',
            palette='Set2', order=['High Value', 'Medium Value', 'Low Value'], ax=axes[1])
axes[1].set_title('CLV by Customer Segment')
axes[1].set_xlabel('Segment')
axes[1].set_ylabel('Estimated CLV (£)')

plt.tight_layout()
plt.savefig('notebooks/clv_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("✅ CLV distribution plot saved")

# ── Save updated RFM with CLV ──
rfm.to_csv('data/rfm.csv', index=False)
print("✅ RFM data updated with CLV column → data/rfm.csv")

# ── Top 10 CLV Customers ──
top_clv = rfm.nlargest(10, 'CLV_Estimated')[['CustomerID', 'Recency', 'Frequency',
                                              'Monetary', 'Segment', 'CLV_Estimated']]
print("\n🏆 Top 10 Highest CLV Customers:")
print(top_clv.to_string(index=False))
