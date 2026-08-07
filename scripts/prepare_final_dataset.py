"""
prepare_final_dataset.py

Encodes categorical features, scales numeric features, and creates a
stratified train/val/test split from the validated dataset. Also saves
feature_info.json and scaler_params.json -- these are REQUIRED at
inference time to process new farmer inputs the same way training
data was processed.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

INPUT_PATH = "data/processed/dataset_validated.csv"
OUTPUT_DIR = "data/final"

NUMERIC_FEATURES = [
    "sand_pct", "clay_pct", "silt_pct", "soil_pH", "nitrogen_ppm",
    "organic_carbon_pct", "bulk_density_gcm3", "soil_moisture_pct",
    "phosphorus_ppm", "potassium_ppm", "temperature_C", "humidity_pct",
    "rainfall_7day_mm", "solar_radiation_MJ", "wind_speed_ms",
    "seed_size_mm", "seed_weight_1000g", "sowing_month", "latitude", "longitude",
]

CATEGORICAL_FEATURES = ["state", "season", "soil_texture_class", "crop_name"]
BINARY_FEATURES = ["rain_expected"]

TARGET_REGRESSION = "optimal_depth_cm"
TARGET_CLASSIFICATION = "depth_class"


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        encoders[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

    df["rain_expected_encoded"] = df["rain_expected"].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["crop_name"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["crop_name"]
    )
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(train_df[NUMERIC_FEATURES])
    val_scaled[NUMERIC_FEATURES] = scaler.transform(val_df[NUMERIC_FEATURES])
    test_scaled[NUMERIC_FEATURES] = scaler.transform(test_df[NUMERIC_FEATURES])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_scaled.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val_scaled.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
    test_scaled.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    feature_info = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "target_regression": TARGET_REGRESSION,
        "target_classification": TARGET_CLASSIFICATION,
        "category_encodings": encoders,
    }
    with open(f"{OUTPUT_DIR}/feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    scaler_params = {
        "feature_order": NUMERIC_FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(f"{OUTPUT_DIR}/scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    print(f"\nSaved train/val/test CSVs, feature_info.json, scaler_params.json -> {OUTPUT_DIR}/")
    print(f"\nCrop distribution in train set:\n{train_df['crop_name'].value_counts()}")


if __name__ == "__main__":
    main()