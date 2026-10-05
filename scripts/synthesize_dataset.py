import pandas as pd
import numpy as np
import random
import os
import sys

sys.path.append(os.path.dirname(__file__))
from crop_database import CROP_DATA, calculate_sowing_depth

SOIL_INPUT_PATH = "data/raw/soil_data_raw.csv"
WEATHER_INPUT_PATH = "data/raw/weather_data_raw.csv"
OUTPUT_PATH = "data/processed/dataset_synthesized.csv"

random.seed(42)
np.random.seed(42)



def classify_texture(sand_pct: float, clay_pct: float) -> str:
    if clay_pct >= 40:
        return "Clay"
    elif clay_pct >= 27:
        return "Clay Loam"
    elif sand_pct >= 70 and clay_pct < 15:
        return "Sandy"
    elif sand_pct >= 50:
        return "Sandy Loam"
    return "Loam"


def estimate_soil_moisture_pct(clay_pct: float, sand_pct: float,
                                organic_carbon: float, rainfall_7day_mm: float) -> float:
    base = 8 + clay_pct * 0.35 + organic_carbon * 0.8 - sand_pct * 0.05
    rain_contribution = min(rainfall_7day_mm * 0.3, 15)
    moisture = base + rain_contribution + random.gauss(0, 1.5)
    return round(max(4, min(45, moisture)), 2)

STATE_NPK_REFERENCE_PPM = {
    "Punjab":         {"phosphorus_ppm_range": (8, 16),  "potassium_ppm_range": (85, 135)},
    "Haryana":        {"phosphorus_ppm_range": (8, 16),  "potassium_ppm_range": (60, 100)},
    "Uttar Pradesh":  {"phosphorus_ppm_range": (6, 22),  "potassium_ppm_range": (45, 130)},
    "Madhya Pradesh": {"phosphorus_ppm_range": (4, 25),  "potassium_ppm_range": (65, 175)},
    "Rajasthan":      {"phosphorus_ppm_range": (5, 12),  "potassium_ppm_range": (95, 170)},
}


def estimate_phosphorus_ppm(organic_carbon: float, clay_pct: float, state: str) -> float:
    p_min, p_max = STATE_NPK_REFERENCE_PPM[state]["phosphorus_ppm_range"]
    oc_score = min(max((organic_carbon - 2) / (12 - 2), 0), 1)
    clay_score = min(max(clay_pct / 60, 0), 1)
    fertility_score = 0.7 * oc_score + 0.3 * clay_score
    base = p_min + fertility_score * (p_max - p_min)
    value = base + random.gauss(0, (p_max - p_min) * 0.08)
    return round(max(p_min * 0.7, min(p_max * 1.3, value)), 2)


def estimate_potassium_ppm(clay_pct: float, sand_pct: float, state: str) -> float:
    k_min, k_max = STATE_NPK_REFERENCE_PPM[state]["potassium_ppm_range"]
    clay_score = min(max(clay_pct / 60, 0), 1)
    sand_penalty = min(max(sand_pct / 100, 0), 1)
    fertility_score = min(max(clay_score - 0.3 * sand_penalty, 0), 1)
    base = k_min + fertility_score * (k_max - k_min)
    value = base + random.gauss(0, (k_max - k_min) * 0.08)
    return round(max(k_min * 0.7, min(k_max * 1.3, value)), 2)


def classify_depth(depth_cm: float) -> str:
    if depth_cm < 3:
        return "Shallow"
    elif depth_cm < 6:
        return "Medium"
    return "Deep"


def main():
    soil_df = pd.read_csv(SOIL_INPUT_PATH)
    weather_df = pd.read_csv(WEATHER_INPUT_PATH)

    merged = weather_df.merge(soil_df, on=["latitude", "longitude", "state"], how="inner")
    print(f"Merged soil + weather: {len(merged)} soil-weather-season combinations.")

    rows = []
    for _, r in merged.iterrows():
        texture_class = classify_texture(r["sand"], r["clay"])
        moisture_pct = estimate_soil_moisture_pct(r["clay"], r["sand"], r["soc"], r["rainfall_7day_mm"])
        phosphorus = estimate_phosphorus_ppm(r["soc"], r["clay"], r["state"])
        potassium = estimate_potassium_ppm(r["clay"], r["sand"], r["state"])
        nitrogen_ppm = round(r["nitrogen"] * 1000, 2)  

        matching_crops = [c for c, d in CROP_DATA.items() if d["season"] == r["season"]]

        for crop_name in matching_crops:
            result = calculate_sowing_depth(
                crop_name=crop_name,
                soil_texture_class=texture_class,
                soil_moisture_pct=moisture_pct,
                rain_expected=bool(r["rain_expected"]),
                rainfall_7day_mm=r["rainfall_7day_mm"],
                temperature_c=r["temperature_C"],
            )

            depth_min, depth_max = result["safe_range_cm"]
            noisy_depth = result["recommended_depth_cm"] + random.gauss(0, 0.15)
            min_allowed = depth_min - 0.3
            max_allowed = depth_max + 0.3
            noisy_depth = round(min(max(noisy_depth, min_allowed), max_allowed), 2)

            rows.append({
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "state": r["state"],
                "season": r["season"],
                "sowing_month": r["sowing_month"],
                "crop_name": crop_name,
                "sand_pct": r["sand"],
                "clay_pct": r["clay"],
                "silt_pct": r["silt"],
                "soil_pH": r["phh2o"],
                "organic_carbon_gkg": r["soc"],
                "bulk_density_gcm3": r["bdod"],
                "nitrogen_ppm": nitrogen_ppm,
                "soil_texture_class": texture_class,
                "soil_moisture_pct": moisture_pct,
                "phosphorus_ppm": phosphorus,
                "potassium_ppm": potassium,
                "npk_source": "measured_N_estimated_PK_calibrated_to_literature",
                "moisture_source": "estimated_heuristic",
                "temperature_C": r["temperature_C"],
                "humidity_pct": r["humidity_pct"],
                "rainfall_7day_mm": r["rainfall_7day_mm"],
                "rain_expected": r["rain_expected"],
                "solar_radiation_MJ": r["solar_radiation_MJ"],
                "wind_speed_ms": r["wind_speed_ms"],
                "seed_size_mm": CROP_DATA[crop_name]["seed_size_mm"],
                "seed_weight_1000g": CROP_DATA[crop_name]["seed_weight_1000g"],
                "optimal_depth_cm": noisy_depth,
                "depth_min_cm": depth_min,
                "depth_max_cm": depth_max,
                "depth_class": classify_depth(noisy_depth),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone. Synthesized {len(df)} training rows -> {OUTPUT_PATH}")
    print(f"\nRows per crop:\n{df['crop_name'].value_counts()}")
    print(f"\nDepth class distribution:\n{df['depth_class'].value_counts()}")
    print(f"\nSample rows:\n{df.head(3)}")


if __name__ == "__main__":
    main()