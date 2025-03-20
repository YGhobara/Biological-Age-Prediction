import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_PATH = "../reports/final_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Step 1: Features & target (chronological age)
X = df.drop(columns=["RIDAGEYR"]) # all predictors/columns EXCLUDING chronological age
y = df["RIDAGEYR"]  # target: chronological age

# Step 2: Splitting data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10], 
    "min_samples_leaf": [1, 2, 4] 
}

# Step 4: Training & grid search to find best parameters
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Step 5: Evaluate the model, testing part
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random forest model trained, MAE: {mae:.2f} years, R²: {r2:.2f}")
print(f"Best parameters for RF: {grid_search.best_params_}")

# Step 6: Saving the model
joblib.dump(best_model, "../models/biological_age_rf.pkl")
print("Best model saved to ../models/biological_age_rf.pkl")
