import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/raw/recs2020.csv")

# Define features
selected_features = [
    'state_postal', 'BA_climate', 'IECC_climate_code',
    'TYPEHUQ', 'YEARMADERANGE', 'BEDROOMS', 'NCOMBATH', 'NHAFBATH',
    'OTHROOMS', 'TOTROOMS', 'TOTSQFT_EN', 'STORIES',
    'NHSLDMEM', 'FUELHEAT', 'NUMFRIG', 'NUMFREEZ',
    'WALLTYPE', 'OVEN'
]

# Prepare features and target
features_df = df[selected_features + ['KWH']].copy()
X = features_df.drop(columns=['KWH'])
y = features_df['KWH']

# One-hot encode categorical features
categorical_cols = [
    'state_postal', 'BA_climate', 'IECC_climate_code',
    'TYPEHUQ', 'YEARMADERANGE', 'FUELHEAT', 'WALLTYPE', 'OVEN'
]
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_model.pkl")
print("Model saved to models/rf_model.pkl")