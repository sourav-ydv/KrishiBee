import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from agrodynamicnet import AgroDynamicNet

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "data/final"
MODELS_DIR = "models"
REPORT_PATH = "data/processed/nn_report.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOIL_FEATURES = [
    "sand_pct", "clay_pct", "silt_pct", "soil_pH", "nitrogen_ppm",
    "organic_carbon_pct", "bulk_density_gcm3", "soil_moisture_pct",
    "phosphorus_ppm", "potassium_ppm", "soil_texture_class_encoded",
]
WEATHER_FEATURES = [
    "temperature_C", "humidity_pct", "rainfall_7day_mm",
    "solar_radiation_MJ", "wind_speed_ms", "rain_expected_encoded",
    "season_encoded", "sowing_month",
]
CROP_NUMERIC_FEATURES = ["seed_size_mm", "seed_weight_1000g", "latitude", "longitude"]
CROP_IDX_COL = "crop_name_encoded"
STATE_IDX_COL = "state_encoded"
TARGET = "optimal_depth_cm"


class DepthDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.soil = torch.tensor(df[SOIL_FEATURES].values, dtype=torch.float32)
        self.weather = torch.tensor(df[WEATHER_FEATURES].values, dtype=torch.float32)
        self.crop_numeric = torch.tensor(df[CROP_NUMERIC_FEATURES].values, dtype=torch.float32)
        self.crop_idx = torch.tensor(df[CROP_IDX_COL].values, dtype=torch.long)
        self.state_idx = torch.tensor(df[STATE_IDX_COL].values, dtype=torch.long)
        self.y = torch.tensor(df[TARGET].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.soil[i], self.weather[i], self.crop_numeric[i],
                self.crop_idx[i], self.state_idx[i], self.y[i])


def load_datasets():
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    with open(f"{DATA_DIR}/feature_info.json") as f:
        feature_info = json.load(f)
    n_crops = len(feature_info["category_encodings"]["crop_name"])
    n_states = len(feature_info["category_encodings"]["state"])
    return train_df, val_df, test_df, n_crops, n_states


def train_one_variant(variant_name, use_soil, use_weather, use_crop,
                       train_ds, val_ds, n_crops, n_states,
                       max_epochs=300, patience=25, batch_size=32, lr=1e-3):
    model = AgroDynamicNet(
        n_soil_features=len(SOIL_FEATURES),
        n_weather_features=len(WEATHER_FEATURES),
        n_crop_numeric_features=len(CROP_NUMERIC_FEATURES),
        n_crops=n_crops,
        n_states=n_states,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\n=== Training variant: {variant_name} ===")
    for epoch in range(max_epochs):
        model.train()
        for soil, weather, crop_num, crop_idx, state_idx, y in train_loader:
            soil, weather, crop_num = soil.to(DEVICE), weather.to(DEVICE), crop_num.to(DEVICE)
            crop_idx, state_idx, y = crop_idx.to(DEVICE), state_idx.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(soil, weather, crop_num, crop_idx, state_idx,
                          use_soil=use_soil, use_weather=use_weather, use_crop=use_crop)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for soil, weather, crop_num, crop_idx, state_idx, y in val_loader:
                soil, weather, crop_num = soil.to(DEVICE), weather.to(DEVICE), crop_num.to(DEVICE)
                crop_idx, state_idx = crop_idx.to(DEVICE), state_idx.to(DEVICE)
                preds = model(soil, weather, crop_num, crop_idx, state_idx,
                              use_soil=use_soil, use_weather=use_weather, use_crop=use_crop)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y.numpy())

        val_mae = mean_absolute_error(val_true, val_preds)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: val_MAE={val_mae:.4f}cm (best={best_val_mae:.4f})")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}. Best val MAE: {best_val_mae:.4f}cm")
            break

    model.load_state_dict(best_state)
    return model, best_val_mae


def evaluate_variant(model, dataset, df, use_soil, use_weather, use_crop, variant_name):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    preds_all = []
    with torch.no_grad():
        for soil, weather, crop_num, crop_idx, state_idx, y in loader:
            soil, weather, crop_num = soil.to(DEVICE), weather.to(DEVICE), crop_num.to(DEVICE)
            crop_idx, state_idx = crop_idx.to(DEVICE), state_idx.to(DEVICE)
            preds = model(soil, weather, crop_num, crop_idx, state_idx,
                          use_soil=use_soil, use_weather=use_weather, use_crop=use_crop)
            preds_all.extend(preds.cpu().numpy())

    df = df.copy()
    df["prediction"] = preds_all
    mae = mean_absolute_error(df[TARGET], df["prediction"])
    rmse = np.sqrt(mean_squared_error(df[TARGET], df["prediction"]))
    r2 = r2_score(df[TARGET], df["prediction"])

    per_crop = []
    for crop_encoded in df[CROP_IDX_COL].unique():
        subset = df[df[CROP_IDX_COL] == crop_encoded]
        if subset[TARGET].std() > 1e-6:
            r2_within = r2_score(subset[TARGET], subset["prediction"])
        else:
            r2_within = float("nan")
        per_crop.append({"crop_encoded": crop_encoded,
                          "mae_cm": round(subset["abs_error"].mean() if "abs_error" in subset else
                                          np.abs(subset[TARGET] - subset["prediction"]).mean(), 3),
                          "r2_within_crop": round(r2_within, 3)})

    print(f"\n  [{variant_name}] Test: MAE={mae:.3f}cm  RMSE={rmse:.3f}cm  R2={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "per_crop": per_crop}


def main():
    print(f"Using device: {DEVICE}")
    train_df, val_df, test_df, n_crops, n_states = load_datasets()

    train_ds = DepthDataset(train_df)
    val_ds = DepthDataset(val_df)
    test_ds = DepthDataset(test_df)

    variants = {
        "combined":      dict(use_soil=True,  use_weather=True,  use_crop=True),
        "soil_only":     dict(use_soil=True,  use_weather=False, use_crop=False),
        "weather_only":  dict(use_soil=False, use_weather=True,  use_crop=False),
        "crop_only":     dict(use_soil=False, use_weather=False, use_crop=True),
    }

    results = {}
    os.makedirs(MODELS_DIR, exist_ok=True)
    report_lines = [f"Device used: {DEVICE}\n"]

    for name, flags in variants.items():
        model, best_val_mae = train_one_variant(
            name, flags["use_soil"], flags["use_weather"], flags["use_crop"],
            train_ds, val_ds, n_crops, n_states,
        )
        test_metrics = evaluate_variant(model, test_ds, test_df,
                                         flags["use_soil"], flags["use_weather"], flags["use_crop"], name)
        results[name] = test_metrics
        torch.save(model.state_dict(), f"{MODELS_DIR}/agrodynamicnet_{name}.pt")
        report_lines.append(f"\n=== {name} ===")
        report_lines.append(f"Val MAE during training: {best_val_mae:.4f}cm")
        report_lines.append(f"Test: MAE={test_metrics['mae']:.4f}cm RMSE={test_metrics['rmse']:.4f}cm R2={test_metrics['r2']:.4f}")

    print("\n\n=== ABLATION SUMMARY (test set) ===")
    print(f"{'Variant':<15} {'MAE (cm)':<12} {'RMSE (cm)':<12} {'R2':<8}")
    for name, m in results.items():
        print(f"{name:<15} {m['mae']:<12.4f} {m['rmse']:<12.4f} {m['r2']:<8.4f}")

    report_lines.append("\n\n=== ABLATION SUMMARY (test set) ===")
    report_lines.append(f"{'Variant':<15} {'MAE (cm)':<12} {'RMSE (cm)':<12} {'R2':<8}")
    for name, m in results.items():
        report_lines.append(f"{name:<15} {m['mae']:<12.4f} {m['rmse']:<12.4f} {m['r2']:<8.4f}")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()