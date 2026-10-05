import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_DIR = "data/final"
MODELS_DIR = "models"
REPORT_PATH = "data/processed/baseline_report.txt"

TARGET = "optimal_depth_cm"

EXCLUDE_COLS = [
    "optimal_depth_cm", "depth_class", "depth_min_cm", "depth_max_cm",
    "state", "season", "soil_texture_class", "crop_name", "rain_expected",
    "npk_source", "moisture_source",
]


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    val = pd.read_csv(f"{DATA_DIR}/val.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, val, test


def get_X_y(df):
    X = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    y = df[TARGET]
    return X, y


def evaluate(model, X, y, label):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"  {label}: MAE={mae:.3f}cm  RMSE={rmse:.3f}cm  R2={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def per_crop_breakdown(model, df, X, y, model_name):
    preds = model.predict(X)
    df = df.copy()
    df["prediction"] = preds
    df["abs_error"] = np.abs(df[TARGET] - df["prediction"])

    rows = []
    for crop in df["crop_name"].unique():
        subset = df[df["crop_name"] == crop]
        mae = subset["abs_error"].mean()
        if subset[TARGET].std() > 1e-6:
            r2_within = r2_score(subset[TARGET], subset["prediction"])
        else:
            r2_within = float("nan")
        rows.append({"crop": crop, "mae_cm": round(mae, 3), "r2_within_crop": round(r2_within, 3)})

    result = pd.DataFrame(rows).sort_values("r2_within_crop")
    print(f"\n  Per-crop breakdown ({model_name}):")
    print(result.to_string(index=False))
    return result


def main():
    train, val, test = load_data()
    X_train, y_train = get_X_y(train)
    X_val, y_val = get_X_y(val)
    X_test, y_test = get_X_y(test)

    print(f"Features used ({len(X_train.columns)}): {list(X_train.columns)}\n")

    report_lines = []
    os.makedirs(MODELS_DIR, exist_ok=True)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
    }

    for name, model in models.items():
        print(f"=== {name} ===")
        report_lines.append(f"\n=== {name} ===")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
        print(f"  5-fold CV MAE on train: {cv_mae.mean():.3f}cm (+/- {cv_mae.std():.3f})")
        report_lines.append(f"5-fold CV MAE on train: {cv_mae.mean():.3f}cm (+/- {cv_mae.std():.3f})")

        model.fit(X_train, y_train)

        val_metrics = evaluate(model, X_val, y_val, "Validation")
        test_metrics = evaluate(model, X_test, y_test, "Test")
        report_lines.append(f"Validation: {val_metrics}")
        report_lines.append(f"Test: {test_metrics}")

        # feature importance
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        importances = importances.sort_values(ascending=False)
        print(f"\n  Top 8 feature importances:")
        print(importances.head(8).to_string())
        report_lines.append(f"\nTop 8 feature importances:\n{importances.head(8).to_string()}")

        crop_breakdown = per_crop_breakdown(model, test, X_test, y_test, name)
        report_lines.append(f"\nPer-crop breakdown:\n{crop_breakdown.to_string(index=False)}")

        joblib.dump(model, f"{MODELS_DIR}/{name.lower()}_baseline.pkl")
        print()

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Full report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()