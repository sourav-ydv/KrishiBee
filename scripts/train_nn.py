import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from agrodynamicnet import AgroDynamicNet

DATA_DIR = "data/final"
MODELS_DIR = "models"
REPORT_DIR = "data/processed"

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
CROP_IDX_COL = "crop_name_encoded"
STATE_IDX_COL = "state_encoded"
TARGET = "optimal_depth_cm"

VARIANTS = {
    "combined": dict(use_soil=True, use_weather=True, use_crop=True),
    "soil_only": dict(use_soil=True, use_weather=False, use_crop=False),
    "weather_only": dict(use_soil=False, use_weather=True, use_crop=False),
    "crop_only": dict(use_soil=False, use_weather=False, use_crop=True),
}


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


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
                       train_ds, val_ds, n_crops, n_states, seed,
                       max_epochs=300, patience=25, batch_size=32, lr=1e-3):
    set_seed(seed)
    model = AgroDynamicNet(
        n_soil_features=len(SOIL_FEATURES),
        n_weather_features=len(WEATHER_FEATURES),
        n_crop_numeric_features=len(CROP_NUMERIC_FEATURES),
        n_crops=n_crops,
        n_states=n_states,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\n=== Training variant: {variant_name} (seed {seed}) ===")
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

    print(f"\n  [{variant_name}] Test: MAE={mae:.3f}cm  RMSE={rmse:.3f}cm  R2={r2:.3f}")
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def write_nn_multiseed_summary():
    rows = []
    for name in os.listdir(REPORT_DIR):
        if name.startswith("nn_metrics_seed_") and name.endswith(".json"):
            with open(os.path.join(REPORT_DIR, name), encoding="utf-8") as f:
                rows.append(json.load(f))
    rows.sort(key=lambda r: r["seed"])
    if not rows:
        return
    variants = list(rows[0]["variants"].keys())
    lines = [f"Seeds: {', '.join(str(r['seed']) for r in rows)}", ""]
    lines.append(f"{'Variant':<16}{'MAE mean':<12}{'MAE std':<12}{'R2 mean':<12}{'R2 std':<12}")
    for name in variants:
        mae_mean, mae_std = mean_std([r["variants"][name]["mae"] for r in rows])
        r2_mean, r2_std = mean_std([r["variants"][name]["r2"] for r in rows])
        lines.append(f"{name:<16}{mae_mean:<12.4f}{mae_std:<12.4f}{r2_mean:<12.4f}{r2_std:<12.4f}")
    lines.append("")
    lines.append("Error reduction of combined versus crop-only, (crop_only MAE - combined MAE) / crop_only MAE")
    reductions = []
    for row in rows:
        crop_mae = row["variants"]["crop_only"]["mae"]
        combined_mae = row["variants"]["combined"]["mae"]
        reduction = (crop_mae - combined_mae) / crop_mae * 100
        reductions.append(reduction)
        lines.append(f"seed {row['seed']}: {reduction:.1f}%")
    red_mean, red_std = mean_std(reductions)
    lines.append(f"mean: {red_mean:.1f}%  std: {red_std:.1f}%")
    lines.append("")
    lines.append("Deployed weights left unchanged: models/agrodynamicnet_combined.pt")
    lines.append("Single-run report left unchanged: data/processed/nn_report.txt")
    path = os.path.join(REPORT_DIR, "nn_multiseed.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nMulti-seed summary saved -> {path}")


def run_seed(seed, train_ds, val_ds, test_ds, test_df, n_crops, n_states):
    print(f"\n########## Seed {seed} ##########")
    results = {}
    report_lines = [f"Seed: {seed}", f"Device used: {DEVICE}", ""]
    for name, flags in VARIANTS.items():
        model, best_val_mae = train_one_variant(
            name, flags["use_soil"], flags["use_weather"], flags["use_crop"],
            train_ds, val_ds, n_crops, n_states, seed,
        )
        test_metrics = evaluate_variant(
            model, test_ds, test_df,
            flags["use_soil"], flags["use_weather"], flags["use_crop"], name,
        )
        results[name] = test_metrics
        weight_path = os.path.join(MODELS_DIR, f"agrodynamicnet_{name}_seed{seed}.pt")
        if os.path.basename(weight_path) == "agrodynamicnet_combined.pt":
            raise RuntimeError("Refusing to overwrite the deployed weights")
        torch.save(model.state_dict(), weight_path)
        print(f"  Saved weights -> {weight_path}")
        report_lines.append(f"=== {name} ===")
        report_lines.append(f"Val MAE during training: {best_val_mae:.4f}cm")
        report_lines.append(
            f"Test: MAE={test_metrics['mae']:.4f}cm RMSE={test_metrics['rmse']:.4f}cm R2={test_metrics['r2']:.4f}"
        )
        report_lines.append("")

    report_lines.append("=== ABLATION SUMMARY (test set) ===")
    report_lines.append(f"{'Variant':<15} {'MAE (cm)':<12} {'RMSE (cm)':<12} {'R2':<8}")
    for name, metrics in results.items():
        report_lines.append(
            f"{name:<15} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f} {metrics['r2']:<8.4f}"
        )
    report_path = os.path.join(REPORT_DIR, f"nn_report_seed_{seed}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    metrics_path = os.path.join(REPORT_DIR, f"nn_metrics_seed_{seed}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "variants": results}, f, indent=2)
    print(f"\nSeed report saved -> {report_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    if not seeds:
        raise SystemExit("Pass at least one seed, for example --seeds 0,1,2,3,4")

    print(f"Using device: {DEVICE}")
    print(f"Seeds: {seeds}")
    print("The deployed file models/agrodynamicnet_combined.pt is not written by this script.")
    train_df, val_df, test_df, n_crops, n_states = load_datasets()
    train_ds = DepthDataset(train_df)
    val_ds = DepthDataset(val_df)
    test_ds = DepthDataset(test_df)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    for seed in seeds:
        run_seed(seed, train_ds, val_ds, test_ds, test_df, n_crops, n_states)
        write_nn_multiseed_summary()


if __name__ == "__main__":
    main()
