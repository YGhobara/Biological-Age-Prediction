import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "../reports/final_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Load stacked model
MODEL_PATH = "../models/stacked_meta_model.pkl"
model = joblib.load(MODEL_PATH)

# Load base models
rf = joblib.load("../models/stacked_random_forest.pkl")
xgb = joblib.load("../models/stacked_xgboost.pkl")

# Features & target
X = df.drop(columns=["RIDAGEYR"])  # Features excluding chronological age
y = df["RIDAGEYR"]  # Chronological age

# Get predictions from base models
rf_preds = rf.predict(X)
xgb_preds = xgb.predict(X)

# Stack predictions to match what the meta-model was trained on
meta_X = np.column_stack((rf_preds, xgb_preds))

# Prediction with the meta-model
y_pred = model.predict(meta_X)

# Model evaluation
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Visualization - Stacked model performance: MAE: {mae:.2f} years, R²: {r2:.2f}")

# Used GPT to help generate the plots better and faster

# 📊 1. Scatter Plot - True Age vs Predicted Age
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred, alpha=0.6)
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.title(f"Predicted Age vs. True Age (MAE: {mae:.2f} years, R²: {r2:.2f})")
plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Perfect prediction line
plt.savefig("../reports/visualization_1_scatter.png")  # Save first plot

# 📊 2. Feature Importance - Use XGBoost to determine important biomarkers
try:
    if xgb:  # Ensure XGBoost model exists
        importances = xgb.feature_importances_
        feature_names = X.columns

        # Sort features by importance
        sorted_indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[sorted_indices][:10], y=np.array(feature_names)[sorted_indices][:10])
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Feature Name")
        plt.title("Top 10 Most Important Features (XGBoost)")
        plt.savefig("../reports/visualization_2_feature_importance.png")  # Save second plot
except Exception as e:
    print(f"⚠️ Feature Importance not available: {e}")

# Aging acceleration (Predicted Age - True Age)
df_copy = df.copy()
df_copy["Age Gap"] = y_pred - y
plt.figure(figsize=(8, 6))
sns.histplot(df_copy["Age Gap"], bins=30, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Predicted Age - True Age (Years)")
plt.ylabel("Frequency")
plt.title("Aging Acceleration Distribution")
plt.savefig("../reports/visualization_3_age_gap.png")  # Save third plot

