import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load the dataset
df = pd.read_csv("data/raw/recs2020.csv")

# Select fewer, user-known features
selected_features = [
    'state_postal', 'BA_climate', 'BEDROOMS',
    'TOTROOMS', 'TOTSQFT_EN', 'NHSLDMEM', 'FUELHEAT'
]

# Ensure the features and target are clean
df = df[selected_features + ['KWH']].dropna()

X = df[selected_features]
y = df['KWH']

# One-hot encode categorical variables
categorical_cols = ['state_postal', 'BA_climate', 'FUELHEAT']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
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