import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


df = pd.read_csv("../reports/final_dataset.csv")
X = df.drop(columns=["RIDAGEYR"])
y = df["RIDAGEYR"]

rf = joblib.load("../models/stacked_random_forest.pkl")
xgb = joblib.load("../models/stacked_xgboost.pkl")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
r2_scores = []

for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
    # Generate meta features using base models
    rf_train_preds = rf.predict(X_train_fold)
    xgb_train_preds = xgb.predict(X_train_fold)
    meta_X_train_fold = np.column_stack((rf_train_preds, xgb_train_preds))
    
    rf_val_preds = rf.predict(X_val_fold)
    xgb_val_preds = xgb.predict(X_val_fold)
    meta_X_val_fold = np.column_stack((rf_val_preds, xgb_val_preds))
    
    # Train a new meta-model on this fold's meta training data
    meta_model_cv = XGBRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.8,
        reg_lambda=5,
        reg_alpha=5,
        random_state=42
    )
    meta_model_cv.fit(meta_X_train_fold, y_train_fold)
    
    # Evaluate on validation meta features
    y_val_pred = meta_model_cv.predict(meta_X_val_fold)
    mae_fold = mean_absolute_error(y_val_fold, y_val_pred)
    r2_fold = r2_score(y_val_fold, y_val_pred)
    mae_scores.append(mae_fold)
    r2_scores.append(r2_fold)

print(f"Cross-Validated MAE: {np.mean(mae_scores):.2f} years (±{np.std(mae_scores):.2f})")
print(f"Cross-Validated R²: {np.mean(r2_scores):.2f} (±{np.std(r2_scores):.2f})")
