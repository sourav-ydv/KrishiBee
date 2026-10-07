# KrishiBee

An ML-powered sowing depth advisory system for Indian farmers. Given a crop, a location, and a sowing season, it recommends a sowing depth based on soil properties, weather conditions, and crop characteristics — and shows its work, rather than just outputting a number.

## The problem

Sowing depth matters more than it gets credit for. Plant a seed too shallow and it dries out or gets eaten; too deep and it never breaks the surface. Most advisory material gives a single fixed number per crop ("sow wheat 4-5cm deep") and ignores that the right depth shifts with soil texture, moisture, and weather. Small and marginal farmers in India mostly work from this kind of generic advice, or none at all.

KrishiBee tries to close that gap: it combines real soil and weather data for a location with agronomic rules and a trained model to give a depth recommendation that actually accounts for local conditions.

## How it works

**Data pipeline.** Soil properties (texture, pH, organic carbon, bulk density, nitrogen) for 145 coordinates across Punjab, Haryana, Uttar Pradesh, Madhya Pradesh, and Rajasthan were pulled from SoilGrids. Weather climatology (temperature, humidity, rainfall, solar radiation) for the same points was pulled from NASA POWER, for both Rabi and Kharif sowing windows. Phosphorus and potassium aren't in SoilGrids, so they're estimated from texture and organic carbon, calibrated against published district-level soil fertility studies for each state rather than left as an unconstrained guess.

**Agronomic rules.** `crop_database.py` encodes base sowing depth ranges for 8 crops (Wheat, Rice, Maize, Chickpea, Mustard, Soybean, Pearl Millet, Pigeon Pea), sourced from ICAR/FAO guidance, along with continuous adjustment functions for soil texture, moisture, rainfall, and temperature. This is both the training-label generator for the synthetic dataset and a standalone fallback the app cross-checks every prediction against.

**Synthesized training data.** Combining the soil and weather data through the rule engine produces 1,160 labeled rows (145 locations × up to 8 season-matched crops), with small added noise so labels aren't perfectly deterministic.

**Models.** Random Forest and XGBoost baselines were trained first, both landing around MAE 0.12-0.13cm / R² 0.99 on held-out test data. A custom multi-branch neural network (separate soil, weather, and crop+location branches merged before the output layer) was trained alongside them, with a 4-way ablation study (soil-only, weather-only, crop-only, combined) to check whether soil and weather features actually carry predictive signal beyond crop identity alone. They do: the combined model cuts error by about 36% relative to crop-only. The tree-based baselines edge out the neural net slightly, consistent with typical small-tabular-data behavior — the app deploys the custom network anyway, since that was the deliberate choice for this project, with the comparison documented rather than hidden.

**Honesty about the data.** Every label in the training set is derived from the rule engine, not from observed real-world outcomes. That means the model's accuracy reflects how well it learned the rules, not how well those rules match reality in the field. Real field validation (actual farms, actual germination outcomes) was out of reach for this build due to access constraints — noted here as a limitation, not hidden.

**API.** FastAPI backend with two prediction modes:
- `/predict` — manual input of all soil/weather values
- `/predict_by_location` — farmer supplies only crop, state, and GPS coordinates; soil and weather are fetched automatically

Both return the neural network's prediction alongside the rule engine's prediction and the agronomic safe range, so a farmer (or a developer debugging) can see when the two disagree. `/predict_by_location` falls back to the nearest previously-collected soil point when the live SoilGrids API is unavailable — ISRIC paused the SoilGrids REST API service during this project's development, so this isn't a hypothetical edge case, it's handling a real outage gracefully.

**Frontend.** React app with a Leaflet map for picking a location, a form for crop/state/season, and a results card showing both predictions, the safe range, and the derived soil texture and moisture.

## Tech stack

- **Data collection:** Python, SoilGrids REST API, NASA POWER API
- **ML:** scikit-learn (Random Forest), XGBoost, PyTorch (custom neural network)
- **Backend:** FastAPI, uvicorn
- **Frontend:** React (Vite), Leaflet, Axios

## Project structure

```
KrishiBee/
├── scripts/          # data collection, synthesis, validation, training
├── models/           # model architecture + trained weights
├── api/               # FastAPI backend
├── frontend/          # React app
├── data/
│   ├── raw/           # SoilGrids + NASA POWER output
│   ├── processed/     # synthesized + validated datasets, reports
│   └── final/          # train/val/test splits, scaler + feature config
└── notebooks/
```

## Running locally

**Backend:**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Results summary

| Model | Test MAE (cm) | Test R² |
|---|---|---|
| Random Forest | 0.127 | 0.989 |
| XGBoost | 0.121 | 0.990 |
| AgroDynamicNet (combined) | 0.153 | 0.985 |
| AgroDynamicNet (crop-only ablation) | 0.208 | 0.972 |

Per-crop performance varies — strongest for Maize and Pearl Millet (R² 0.9+ within-crop), weaker for Mustard, which has the narrowest allowed depth range and so the least room for soil/weather features to move the prediction.

## Known limitations

- Training labels come from the rule engine, not observed field outcomes
- Phosphorus, potassium, and soil moisture are estimated, not measured
- No real-world field validation yet (documented as future work, not silently skipped)
- SoilGrids live lookups can fall back to a cached nearby point when the external API is down

## Future work

- Real field data collection for model validation against actual outcomes
- RAG-based chatbot for broader farming advisory (irrigation timing, fertilizer scheduling)
- Expand beyond the current 8 crops and 5 states

## Author

Sourav Yadav