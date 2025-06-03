from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "CT": 0.3255, "ME": 0.2827, "MA": 0.3019, "NH": 0.2281, "RI": 0.3230, "VT": 0.2259,
    "NJ": 0.1988, "NY": 0.2543, "PA": 0.1843, "IL": 0.1759, "IN": 0.1652, "MI": 0.1937,
    "OH": 0.1612, "WI": 0.1781, "IA": 0.1255, "KS": 0.1429, "MN": 0.1512, "MO": 0.1197,
    "NE": 0.1174, "ND": 0.1108, "SD": 0.1275, "DE": 0.1670, "DC": 0.2040, "FL": 0.1509,
    "GA": 0.1471, "MD": 0.1895, "NC": 0.1476, "SC": 0.1518, "VA": 0.1502, "WV": 0.1589,
    "AL": 0.1650, "KY": 0.1341, "MS": 0.1451, "TN": 0.1351, "AR": 0.1256, "LA": 0.1294,
    "OK": 0.1237, "TX": 0.1530, "AZ": 0.1519, "CO": 0.1514, "ID": 0.1156, "MT": 0.1196,
    "NV": 0.1447, "NM": 0.1487, "UT": 0.1241, "WY": 0.1242, "CA": 0.3241, "OR": 0.1513,
    "WA": 0.1263, "AK": 0.2579, "HI": 0.4111
    }
    rate = state_rates.get(data.state_postal.upper(), 0.1711)  # fallback to U.S. avg
    estimated_cost = round(predicted_kwh * rate, 2)

    return {
        "predicted_kwh": round(predicted_kwh, 2),
        "estimated_cost_usd": estimated_cost,
        "rate_used": rate
    }