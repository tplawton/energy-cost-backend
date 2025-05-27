from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load("models/rf_model.pkl")

# Define input schema (based on selected features)
class EnergyInput(BaseModel):
    state_postal: str
    BA_climate: str
    IECC_climate_code: str
    TYPEHUQ: str
    YEARMADERANGE: str
    BEDROOMS: int
    NCOMBATH: float
    NHAFBATH: float
    OTHROOMS: float
    TOTROOMS: float
    TOTSQFT_EN: float
    STORIES: float
    NHSLDMEM: int
    FUELHEAT: str
    NUMFRIG: int
    NUMFREEZ: int
    WALLTYPE: str
    OVEN: str

app = FastAPI()

@app.post("/predict")
def predict_energy(data: EnergyInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # One-hot encode using the same logic as training
    categorical_cols = [
        'state_postal', 'BA_climate', 'IECC_climate_code',
        'TYPEHUQ', 'YEARMADERANGE', 'FUELHEAT', 'WALLTYPE', 'OVEN'
    ]
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols)

    # Align input features with training columns
    model_input_cols = model.feature_names_in_
    for col in model_input_cols:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[model_input_cols]

    # Make prediction
    predicted_kwh = model.predict(input_df_encoded)[0]

    # Estimate cost using simple rate dictionary
    state_rates = {
        'MA': 0.23, 'TX': 0.14, 'CA': 0.25, 'NY': 0.22, 'FL': 0.16
    }
    rate = state_rates.get(data.state_postal.upper(), 0.18)  # default
    estimated_cost = round(predicted_kwh * rate, 2)

    return {
        "predicted_kwh": round(predicted_kwh, 2),
        "estimated_cost_usd": estimated_cost,
        "rate_used": rate
    }