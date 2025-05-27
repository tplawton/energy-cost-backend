# Energy Cost Predictor

This project predicts household annual electricity consumption (in kWh) using the RECS 2020 dataset and translates it into estimated cost based on user-provided inputs and state electricity rates.

## Project Structure

- `data/raw/` — original RECS dataset (`recs2020.csv`)
- `data/processed/` — preprocessed features and target CSVs (optional)
- `models/` — trained ML model (e.g., `rf_model.pkl`)
- `src/` — data processing and prediction utilities
- `api/` — FastAPI app for deployment
- `notebooks/` — Jupyter notebooks for EDA and experimentation