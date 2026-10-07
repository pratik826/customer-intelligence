import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw_data.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Promotion'] = df['Promotion'].fillna("No Promotion")

# =========================
# TIME-BASED SPLIT
# =========================
split_date = df['Date'].quantile(0.7)

past_df = df[df['Date'] <= split_date]
future_df = df[df['Date'] > split_date]

# =========================
# FEATURE ENGINEERING (PAST)
# =========================
customer_past = past_df.groupby('Customer_Name').agg({
    'Total_Cost': ['sum', 'mean'],
    'Total_Items': 'mean',
    'Transaction_ID': 'count',
    'Date': 'max',
    'Discount_Applied': 'mean'
})

customer_past.columns = [
    'total_spent',
    'avg_spent',
    'avg_items',
    'purchase_count',
    'last_purchase_date',
    'discount_usage'
]

customer_past = customer_past.reset_index()

reference_date = past_df['Date'].max()
customer_past['recency'] = (reference_date - customer_past['last_purchase_date']).dt.days

# =========================
# TARGET CREATION (FUTURE)
# =========================
customer_future = future_df.groupby('Customer_Name').agg({
    'Total_Cost': 'sum'
}).reset_index()

customer_future.columns = ['Customer_Name', 'future_spent']

# merge
customer_df = pd.merge(customer_past, customer_future, on='Customer_Name', how='left')
customer_df['future_spent'] = customer_df['future_spent'].fillna(0)

# define high-value based on FUTURE spending
threshold = customer_df['future_spent'].quantile(0.75)
customer_df['high_value'] = (customer_df['future_spent'] >= threshold).astype(int)

# =========================
# GMM SEGMENTATION (ONLY FOR ANALYSIS)
# =========================
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

features_gmm = customer_df[
    ['total_spent', 'avg_spent', 'avg_items', 'purchase_count', 'recency', 'discount_usage']
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_gmm)

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

customer_df['cluster'] = gmm.predict(X_scaled)

print("\nCluster Summary:")
print(customer_df.groupby('cluster').mean(numeric_only=True))

# =========================
# XGBOOST MODEL (PREDICTION)
# =========================
X = customer_df[
    ['avg_spent', 'avg_items', 'purchase_count', 'recency', 'discount_usage']
]

y = customer_df['high_value']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# compute imbalance weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# =========================
# SHAP EXPLAINABILITY
# =========================
import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# summary plot
shap.summary_plot(shap_values, X_test)

# =========================
# FEATURE IMPORTANCE (fallback)
# =========================
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")