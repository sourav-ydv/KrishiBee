"""
synthesize_dataset.py

Combines real soil data (SoilGrids) + real weather climatology (NASA
POWER) + agronomic depth rules (crop_database.py) into a labeled
training dataset.

IMPORTANT — data provenance:
  MEASURED (real, from APIs):    sand/clay/silt %, pH, organic carbon,
                                  bulk density, nitrogen, temperature,
                                  humidity, rainfall proxy, solar radiation
  ESTIMATED (heuristic, derived): soil_moisture_pct, phosphorus_ppm,
                                  potassium_ppm — SoilGrids does not
                                  provide these; they're derived from
                                  documented soil-science relationships
                                  (texture/organic-carbon correlations)
                                  with added noise, NOT measured values.
  RULE-DERIVED (not measured):   the depth label itself, from
                                  crop_database.py's agronomic rules,
                                  with small random noise added.

This provenance split MUST be stated in the project writeup/methodology
section — do not present estimated columns as if they were measured.
"""

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
    """Simplified USDA-style texture classification (5 buckets, not the full triangle)."""
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
    """
    Heuristic: clay + organic matter hold water (raise moisture),
    sand drains fast (lowers it), recent rainfall adds to it.
    Coefficients are illustrative, not from a specific published study —
    document as an estimation method, not a measured value.
    """
    base = 8 + clay_pct * 0.35 + organic_carbon * 0.8 - sand_pct * 0.05
    rain_contribution = min(rainfall_7day_mm * 0.3, 15)
    moisture = base + rain_contribution + random.gauss(0, 1.5)
    return round(max(4, min(45, moisture)), 2)


def estimate_phosphorus_ppm(organic_carbon: float, clay_pct: float) -> float:
    """Rough positive link to organic carbon, slight negative link to clay (P fixation)."""
    base = 8 + organic_carbon * 3.5 - clay_pct * 0.05 + random.gauss(0, 3)
    return round(max(3, min(60, base)), 2)


def estimate_potassium_ppm(clay_pct: float, sand_pct: float) -> float:
    """Clay minerals hold more exchangeable K than sandy soils."""
    base = 80 + clay_pct * 4 - sand_pct * 1.5 + random.gauss(0, 20)
    return round(max(40, min(500, base)), 2)


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
        phosphorus = estimate_phosphorus_ppm(r["soc"], r["clay"])
        potassium = estimate_potassium_ppm(r["clay"], r["sand"])
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
                "organic_carbon_pct": r["soc"],
                "bulk_density_gcm3": r["bdod"],
                "nitrogen_ppm": nitrogen_ppm,
                "soil_texture_class": texture_class,
                "soil_moisture_pct": moisture_pct,
                "phosphorus_ppm": phosphorus,
                "potassium_ppm": potassium,
                "npk_source": "measured_N_estimated_PK",
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