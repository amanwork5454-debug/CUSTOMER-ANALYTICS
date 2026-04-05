"""
05_churn.py — Customer Churn Prediction

Business Problem:
    Identify customers at risk of churning so marketing teams can
    launch targeted re-engagement campaigns before it's too late.

Definition of Churn:
    A customer is considered churned if they have NOT made a purchase
    in the last 90 days (Recency > 90).

Approach:
    1. Derive churn labels from RFM data
    2. Train and compare Logistic Regression and Random Forest classifiers
    3. Evaluate with Accuracy, ROC-AUC, Precision, Recall, F1
    4. Plot Confusion Matrix, ROC Curve, and Feature Importance
    5. Save the best classifier for deployment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)

# ── Load RFM data ──
rfm = pd.read_csv('data/rfm.csv')
print("RFM shape:", rfm.shape)
print(rfm.head())

# ── Define Churn Label ──
# Customers with Recency > 90 days are considered churned
CHURN_THRESHOLD = 90
rfm['Churned'] = (rfm['Recency'] > CHURN_THRESHOLD).astype(int)

churn_rate = rfm['Churned'].mean()
print(f"\nChurn threshold : {CHURN_THRESHOLD} days")
print(f"Churned customers: {rfm['Churned'].sum()} / {len(rfm)} ({churn_rate:.1%})")

# NOTE: Because the churn label is derived directly from Recency, the classifier
# will achieve near-perfect performance (it re-learns the threshold).
# In a production setting you would add richer features (return rates, product
# categories, session data) to make the prediction genuinely non-trivial.

# ── Features & Target ──
features = ['Recency', 'Frequency', 'Monetary']
X = rfm[features]
y = rfm['Churned']

# ── Scale features ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train/Test Split (stratified to preserve class ratio) ──
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ── Train Models ──
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    y_prob = m.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cv_scores = cross_val_score(m, X_scaled, y, cv=5, scoring='roc_auc')

    results[name] = {
        'model': m, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'auc': auc,
        'cv_auc_mean': cv_scores.mean(), 'cv_auc_std': cv_scores.std()
    }

    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  CV AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))

# ── Best model by CV AUC ──
best_name = max(results, key=lambda x: results[x]['cv_auc_mean'])
best = results[best_name]
print(f"\n✅ Best Model: {best_name} (CV AUC = {best['cv_auc_mean']:.4f})")

# ── Confusion Matrix ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Active', 'Churned'],
                yticklabels=['Active', 'Churned'])
    ax.set_title(f'Confusion Matrix — {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('notebooks/churn_confusion_matrix.png')
plt.show()
print("✅ Confusion matrix saved")

# ── ROC Curve ──
plt.figure(figsize=(8, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Churn Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/churn_roc_curve.png')
plt.show()
print("✅ ROC curve saved")

# ── Feature Importance (Random Forest) ──
rf_model = results['Random Forest']['model']
importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6, 3))
importance.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Importance — Churn (Random Forest)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('notebooks/churn_feature_importance.png')
plt.show()
print("✅ Feature importance saved")

# ── Save Best Churn Model ──
churn_model_data = {
    'model':      best['model'],
    'scaler':     scaler,
    'features':   features,
    'model_name': best_name,
    'accuracy':   best['accuracy'],
    'auc':        best['auc'],
    'cv_auc':     best['cv_auc_mean'],
    'threshold':  CHURN_THRESHOLD
}
with open('model_churn.pkl', 'wb') as f:
    pickle.dump(churn_model_data, f)
print("✅ Churn model saved as model_churn.pkl")

# ── Churn Risk Summary ──
rfm['ChurnProb'] = best['model'].predict_proba(X_scaled)[:, 1]
high_risk = rfm[rfm['ChurnProb'] >= 0.7][['CustomerID', 'Recency', 'Frequency', 'Monetary', 'ChurnProb']]
high_risk = high_risk.sort_values('ChurnProb', ascending=False)
print(f"\n🚨 High-risk customers (churn prob ≥ 70%): {len(high_risk)}")
print(high_risk.head(10).to_string(index=False))
high_risk.to_csv('data/high_risk_customers.csv', index=False)
print("✅ High-risk customers saved to data/high_risk_customers.csv")
