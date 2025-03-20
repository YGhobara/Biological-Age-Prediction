import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_PATH = "../reports/final_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Step 1: Features & target (chronological age)
X = df.drop(columns=["RIDAGEYR"])
y = df["RIDAGEYR"]  # target: chronological age

# Step 2: Splitting data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Training base models (Re-training RF & XGBoost with updated features)
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, min_samples_split=5, random_state=42)
xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.7, colsample_bytree=0.8, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Step 4: Saving base models
joblib.dump(rf, "../models/stacked_random_forest.pkl")
joblib.dump(xgb, "../models/stacked_xgboost.pkl")
print("Updated base models saved (Random Forest & XGBoost)")

# Step 5: Stack/meta model predictions for training
meta_X_train = np.column_stack((rf.predict(X_train), xgb.predict(X_train)))
meta_X_test = np.column_stack((rf.predict(X_test), xgb.predict(X_test)))

# Split training data further for meta model training
meta_X_train, meta_X_val, meta_y_train, meta_y_val = train_test_split(meta_X_train, y_train, test_size=0.2, random_state=42)

# Train meta-model (XGBoost) only on training data
meta_model = XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=3, subsample=0.7, colsample_bytree=0.8, reg_lambda=5, reg_alpha=5, random_state=42)
meta_model.fit(meta_X_train, meta_y_train)

# Step 6: Final predictions using meta-nodel
final_preds = meta_model.predict(meta_X_test)

# Step 7: Model evaluation
mae = mean_absolute_error(y_test, final_preds)
r2 = r2_score(y_test, final_preds)

print(f"Stacking model trained, MAE: {mae:.2f} years, R²: {r2:.2f}")

# Save meta model
joblib.dump(meta_model, "../models/stacked_meta_model.pkl")
print("Stacked model saved successfully")

# Step 8: Check performance on training data
y_train_pred = meta_model.predict(np.column_stack((rf.predict(X_train), xgb.predict(X_train))))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Stacked model performance on training data: MAE: {train_mae:.2f}, R²: {train_r2}")
