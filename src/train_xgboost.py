import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_PATH = "../reports/final_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Step 1: Features & target (chronological age)
X = df.drop(columns=["RIDAGEYR"])
y = df["RIDAGEYR"]  # target: chronological age

# Step 2: Splitting data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],  # Number of boosting rounds
    "learning_rate": [0.01, 0.05, 0.1],  # Step size
    "max_depth": [3, 6, 10],  # Maximum depth of trees
    "subsample": [0.8, 1.0],  # Percentage of training samples used per tree
    "colsample_bytree": [0.8, 1.0]  # Percentage of features used per tree
}

# Step 4: Training & grid search to find best parameters
xgb = XGBRegressor(objective="reg:squarederror", random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

# Step 5: Evaluate the model, testing part
y_pred = best_xgb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"XGBoost model trained, MAE: {mae:.2f} years, R²: {r2:.2f}")
print(f"Best parameters for XGBoost: {grid_search.best_params_}")

# Step 6: Saving the model
joblib.dump(best_xgb, "../models/biological_age_xgboost.pkl")
print("Best XGBoost model saved to ../models/biological_age_xgboost.pkl")
