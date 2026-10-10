import sys
import os
import json
import torch
import random
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from crop_database import CROP_DATA, calculate_sowing_depth
from synthesize_dataset import (
    classify_texture, estimate_soil_moisture_pct,
    estimate_phosphorus_ppm, estimate_potassium_ppm,
)
from collect_soil_data import fetch_soil_point, STATE_BOUNDS
from collect_weather_data import fetch_climatology_point, parse_season_rows, SOWING_MONTHS
from agrodynamicnet import AgroDynamicNet

FEATURE_INFO_PATH = "data/final/feature_info.json"
SCALER_PATH = "data/final/scaler_params.json"
MODEL_PATH = "models/agrodynamicnet_combined.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOIL_FEATURES = [
    "sand_pct", "clay_pct", "silt_pct", "soil_pH", "nitrogen_ppm",
    "organic_carbon_gkg", "bulk_density_gcm3", "soil_moisture_pct",
    "phosphorus_ppm", "potassium_ppm", "soil_texture_class_encoded",
]
WEATHER_FEATURES = [
    "temperature_C", "humidity_pct", "rainfall_7day_mm",
    "solar_radiation_MJ", "wind_speed_ms", "rain_expected_encoded",
    "season_encoded", "sowing_month",
]
CROP_NUMERIC_FEATURES = ["seed_size_mm", "seed_weight_1000g", "latitude", "longitude"]

app = FastAPI(title="KrishiBee Sowing Depth API")

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(FEATURE_INFO_PATH) as f:
    feature_info = json.load(f)
with open(SCALER_PATH) as f:
    scaler_params = json.load(f)

CATEGORY_ENCODINGS = feature_info["category_encodings"]
SCALER_MEAN = dict(zip(scaler_params["feature_order"], scaler_params["mean"]))
SCALER_SCALE = dict(zip(scaler_params["feature_order"], scaler_params["scale"]))

n_crops = len(CATEGORY_ENCODINGS["crop_name"])
n_states = len(CATEGORY_ENCODINGS["state"])

model = AgroDynamicNet(
    n_soil_features=len(SOIL_FEATURES),
    n_weather_features=len(WEATHER_FEATURES),
    n_crop_numeric_features=len(CROP_NUMERIC_FEATURES),
    n_crops=n_crops,
    n_states=n_states,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()


class DepthRequest(BaseModel):
    crop_name: str = Field(..., example="Wheat")
    state: str = Field(..., example="Punjab")
    latitude: float = Field(..., example=30.7)
    longitude: float = Field(..., example=75.8)
    season: str = Field(..., example="Rabi")
    sowing_month: int = Field(..., ge=1, le=12, example=11)

    sand_pct: float = Field(..., ge=0, le=100)
    clay_pct: float = Field(..., ge=0, le=100)
    soil_pH: float = Field(..., ge=3.5, le=10)
    organic_carbon_gkg: float = Field(..., ge=0, le=30)
    bulk_density_gcm3: float = Field(..., ge=0.5, le=2.0)
    nitrogen_ppm: float = Field(..., ge=0)

    temperature_C: float
    humidity_pct: float = Field(..., ge=0, le=100)
    rainfall_7day_mm: float = Field(..., ge=0)
    rain_expected: bool
    solar_radiation_MJ: float = Field(..., ge=0)
    wind_speed_ms: float = Field(..., ge=0)


class LocationDepthRequest(BaseModel):
    crop_name: str = Field(..., example="Wheat")
    state: str = Field(..., example="Punjab")
    latitude: float = Field(..., example=30.7)
    longitude: float = Field(..., example=75.8)
    season: str = Field(..., example="Rabi")


class DepthResponse(BaseModel):
    crop: str
    nn_recommended_depth_cm: float
    rule_based_depth_cm: float
    agronomic_safe_range_cm: list
    within_safe_range: bool
    derived_soil_texture_class: str
    estimated_soil_moisture_pct: float
    note: str


def build_response(crop_name, state, latitude, longitude, season, sowing_month,
                    sand_pct, clay_pct, soil_pH, organic_carbon_gkg, bulk_density_gcm3,
                    nitrogen_ppm, temperature_C, humidity_pct, rainfall_7day_mm,
                    rain_expected, solar_radiation_MJ, wind_speed_ms):
    if crop_name not in CROP_DATA:
        raise HTTPException(400, f"Unknown crop '{crop_name}'. Valid: {list(CROP_DATA.keys())}")
    if state not in CATEGORY_ENCODINGS["state"]:
        raise HTTPException(400, f"Unknown state '{state}'. Valid: {list(CATEGORY_ENCODINGS['state'].keys())}")

    silt_pct = round(100 - sand_pct - clay_pct, 2)
    if silt_pct < 0:
        raise HTTPException(400, "sand_pct + clay_pct exceeds 100%.")

    random.seed(42)
    texture_class = classify_texture(sand_pct, clay_pct)
    moisture_pct = estimate_soil_moisture_pct(clay_pct, sand_pct, organic_carbon_gkg, rainfall_7day_mm)
    phosphorus = estimate_phosphorus_ppm(organic_carbon_gkg, clay_pct, state)
    potassium = estimate_potassium_ppm(clay_pct, sand_pct, state)

    crop = CROP_DATA[crop_name]

    raw = {
        "sand_pct": sand_pct, "clay_pct": clay_pct, "silt_pct": silt_pct,
        "soil_pH": soil_pH, "nitrogen_ppm": nitrogen_ppm,
        "organic_carbon_gkg": organic_carbon_gkg, "bulk_density_gcm3": bulk_density_gcm3,
        "soil_moisture_pct": moisture_pct, "phosphorus_ppm": phosphorus, "potassium_ppm": potassium,
        "temperature_C": temperature_C, "humidity_pct": humidity_pct,
        "rainfall_7day_mm": rainfall_7day_mm, "solar_radiation_MJ": solar_radiation_MJ,
        "wind_speed_ms": wind_speed_ms, "sowing_month": sowing_month,
        "seed_size_mm": crop["seed_size_mm"], "seed_weight_1000g": crop["seed_weight_1000g"],
        "latitude": latitude, "longitude": longitude,
    }

    def scale(name):
        return (raw[name] - SCALER_MEAN[name]) / SCALER_SCALE[name]

    scaled = {k: scale(k) for k in raw if k in SCALER_MEAN}

    soil_texture_encoded = CATEGORY_ENCODINGS["soil_texture_class"].get(texture_class)
    if soil_texture_encoded is None:
        raise HTTPException(500, f"Derived texture class '{texture_class}' not in training encodings.")

    soil_vec = [scaled[f] if f in scaled else soil_texture_encoded for f in SOIL_FEATURES]
    weather_vec = [
        scaled["temperature_C"], scaled["humidity_pct"], scaled["rainfall_7day_mm"],
        scaled["solar_radiation_MJ"], scaled["wind_speed_ms"], int(rain_expected),
        CATEGORY_ENCODINGS["season"][season], scaled["sowing_month"],
    ]
    crop_numeric_vec = [scaled["seed_size_mm"], scaled["seed_weight_1000g"], scaled["latitude"], scaled["longitude"]]
    crop_idx = CATEGORY_ENCODINGS["crop_name"][crop_name]
    state_idx = CATEGORY_ENCODINGS["state"][state]

    soil_t = torch.tensor([soil_vec], dtype=torch.float32).to(DEVICE)
    weather_t = torch.tensor([weather_vec], dtype=torch.float32).to(DEVICE)
    crop_num_t = torch.tensor([crop_numeric_vec], dtype=torch.float32).to(DEVICE)
    crop_idx_t = torch.tensor([crop_idx], dtype=torch.long).to(DEVICE)
    state_idx_t = torch.tensor([state_idx], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        nn_pred = model(soil_t, weather_t, crop_num_t, crop_idx_t, state_idx_t,
                         use_soil=True, use_weather=True, use_crop=True).item()
    nn_pred = round(nn_pred, 2)

    rule_result = calculate_sowing_depth(
        crop_name=crop_name,
        soil_texture_class=texture_class,
        soil_moisture_pct=moisture_pct,
        rain_expected=rain_expected,
        rainfall_7day_mm=rainfall_7day_mm,
        temperature_c=temperature_C,
    )
    safe_min, safe_max = rule_result["safe_range_cm"]
    within_range = safe_min - 0.5 <= nn_pred <= safe_max + 0.5

    note = "Prediction within agronomic safe range." if within_range else \
        "WARNING: NN prediction falls outside the expected agronomic range — treat with caution."

    return DepthResponse(
        crop=crop_name,
        nn_recommended_depth_cm=nn_pred,
        rule_based_depth_cm=rule_result["recommended_depth_cm"],
        agronomic_safe_range_cm=[safe_min, safe_max],
        within_safe_range=within_range,
        derived_soil_texture_class=texture_class,
        estimated_soil_moisture_pct=moisture_pct,
        note=note,
    )

SOIL_CACHE_PATH = "data/raw/soil_data_raw.csv"
_soil_cache = pd.read_csv(SOIL_CACHE_PATH) if os.path.exists(SOIL_CACHE_PATH) else None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def nearest_cached_soil_point(lat, lon):
    if _soil_cache is None or _soil_cache.empty:
        return None
    distances = _soil_cache.apply(
        lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
    )
    nearest_idx = distances.idxmin()
    row = _soil_cache.loc[nearest_idx]
    return {
        "sand": row["sand"], "clay": row["clay"], "phh2o": row["phh2o"],
        "soc": row["soc"], "bdod": row["bdod"], "nitrogen": row["nitrogen"],
    }, round(distances[nearest_idx], 1)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model": "AgroDynamicNet (combined)"}


@app.get("/crops")
def list_crops():
    return {"crops": list(CROP_DATA.keys())}


@app.post("/predict", response_model=DepthResponse)
def predict(req: DepthRequest):
    return build_response(
        req.crop_name, req.state, req.latitude, req.longitude, req.season, req.sowing_month,
        req.sand_pct, req.clay_pct, req.soil_pH, req.organic_carbon_gkg, req.bulk_density_gcm3,
        req.nitrogen_ppm, req.temperature_C, req.humidity_pct, req.rainfall_7day_mm,
        req.rain_expected, req.solar_radiation_MJ, req.wind_speed_ms,
    )


@app.post("/predict_by_location", response_model=DepthResponse)
def predict_by_location(req: LocationDepthRequest):
    if req.season not in SOWING_MONTHS:
        raise HTTPException(400, f"Unknown season '{req.season}'. Valid: {list(SOWING_MONTHS.keys())}")

    if req.state not in STATE_BOUNDS:
        raise HTTPException(400, f"Unknown state '{req.state}'.")
    min_lat, max_lat, min_lon, max_lon = STATE_BOUNDS[req.state]
    if not (min_lat <= req.latitude <= max_lat and min_lon <= req.longitude <= max_lon):
        raise HTTPException(
            400,
            f"That location doesn't fall within {req.state}. Click a point inside {req.state}, "
            f"or change the State dropdown to match where you clicked.",
        )

    soil_data = fetch_soil_point(req.latitude, req.longitude, retries=1, timeout=6)
    required_soil_fields = ["sand", "clay", "phh2o", "soc", "bdod", "nitrogen"]
    missing = [f for f in required_soil_fields if not soil_data or soil_data.get(f) is None]

    used_fallback = False
    fallback_distance_km = None
    if missing:
        fallback = nearest_cached_soil_point(req.latitude, req.longitude)
        if fallback is None:
            raise HTTPException(
                502,
                "Live soil data is unavailable and no cached data could be found. Try the manual /predict endpoint instead.",
            )
        soil_data, fallback_distance_km = fallback
        used_fallback = True

    climatology = fetch_climatology_point(req.latitude, req.longitude)
    if climatology is None:
        raise HTTPException(502, "Could not fetch weather data for this location. Try again shortly.")

    season_rows = parse_season_rows(climatology, req.latitude, req.longitude, req.state)
    weather_row = next((r for r in season_rows if r["season"] == req.season), None)
    if weather_row is None:
        raise HTTPException(502, "Weather data parsing failed for the requested season.")

    result = build_response(
        req.crop_name, req.state, req.latitude, req.longitude, req.season, weather_row["sowing_month"],
        soil_data["sand"], soil_data["clay"], soil_data["phh2o"], soil_data["soc"],
        soil_data["bdod"], round(soil_data["nitrogen"] * 1000, 2),
        weather_row["temperature_C"], weather_row["humidity_pct"], weather_row["rainfall_7day_mm"],
        weather_row["rain_expected"], weather_row["solar_radiation_MJ"], weather_row["wind_speed_ms"],
    )
    if used_fallback:
        result.note = (
            f"Live soil data unavailable (SoilGrids API is currently paused). "
            f"Used cached data from {fallback_distance_km} km away instead. {result.note}"
        )
    return result