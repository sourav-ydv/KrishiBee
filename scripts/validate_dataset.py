"""
validate_dataset.py

Validates the synthesized dataset for structural integrity AND checks
whether the depth label actually varies meaningfully within each crop
(not just between crops) -- this second check matters because if it
doesn't, soil/weather features aren't adding real signal to the target.
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/processed/dataset_synthesized.csv"
OUTPUT_PATH = "data/processed/dataset_validated.csv"
REPORT_PATH = "data/processed/validation_report.txt"
PLOTS_DIR = "data/processed/plots"


def run_structural_checks(df: pd.DataFrame) -> list:
    """Basic sanity checks. Returns list of (check_name, pass/fail, detail) tuples."""
    checks = []

    texture_sum = df["sand_pct"] + df["clay_pct"] + df["silt_pct"]
    bad_texture = ((texture_sum < 98) | (texture_sum > 102)).sum()
    checks.append(("Texture sums to ~100%", bad_texture == 0,
                    f"{bad_texture} rows outside [98,102]% sum"))

    bad_ph = ((df["soil_pH"] < 3.5) | (df["soil_pH"] > 10)).sum()
    checks.append(("pH in valid range (3.5-10)", bad_ph == 0, f"{bad_ph} rows out of range"))

    within_bounds = ((df["optimal_depth_cm"] >= df["depth_min_cm"] - 0.35) &
                      (df["optimal_depth_cm"] <= df["depth_max_cm"] + 0.35))
    checks.append(("Depth within crop bounds (+/- overshoot)", within_bounds.all(),
                    f"{(~within_bounds).sum()} rows out of bounds"))

    missing = df.isnull().sum().sum()
    checks.append(("No missing values", missing == 0, f"{missing} missing cells total"))

    dupes = df.duplicated().sum()
    checks.append(("No exact duplicate rows", dupes == 0, f"{dupes} duplicate rows"))

    bad_moisture = ((df["soil_moisture_pct"] < 3) | (df["soil_moisture_pct"] > 50)).sum()
    checks.append(("Soil moisture in plausible range", bad_moisture == 0,
                    f"{bad_moisture} rows out of range"))

    return checks


def check_within_crop_variance(df: pd.DataFrame) -> pd.DataFrame:
    """
    THE IMPORTANT CHECK: for each crop, how much does depth actually
    vary across different soil/weather conditions? If std is tiny
    relative to the crop's allowed range, soil/weather aren't
    contributing much signal for that crop.
    """
    rows = []
    for crop in df["crop_name"].unique():
        subset = df[df["crop_name"] == crop]
        depth_std = subset["optimal_depth_cm"].std()
        depth_range = subset["depth_max_cm"].iloc[0] - subset["depth_min_cm"].iloc[0]
        pct_of_range = round((depth_std / depth_range) * 100, 1) if depth_range > 0 else 0
        rows.append({
            "crop": crop,
            "depth_std": round(depth_std, 3),
            "crop_allowed_range_cm": depth_range,
            "std_as_pct_of_range": pct_of_range,
        })
    return pd.DataFrame(rows).sort_values("std_as_pct_of_range")


def generate_plots(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    df["optimal_depth_cm"].hist(bins=30, ax=axes[0, 0])
    axes[0, 0].set_title("Depth Distribution (all crops)")
    axes[0, 0].set_xlabel("Depth (cm)")

    df.boxplot(column="optimal_depth_cm", by="crop_name", ax=axes[0, 1], rot=45)
    axes[0, 1].set_title("Depth by Crop (this is the key plot)")
    plt.suptitle("")

    numeric_cols = ["sand_pct", "clay_pct", "soil_pH", "soil_moisture_pct",
                     "temperature_C", "rainfall_7day_mm", "optimal_depth_cm"]
    corr = df[numeric_cols].corr()
    im = axes[1, 0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 0].set_xticks(range(len(numeric_cols)))
    axes[1, 0].set_yticks(range(len(numeric_cols)))
    axes[1, 0].set_xticklabels(numeric_cols, rotation=90, fontsize=8)
    axes[1, 0].set_yticklabels(numeric_cols, fontsize=8)
    axes[1, 0].set_title("Feature Correlation Heatmap")
    plt.colorbar(im, ax=axes[1, 0])

    df["depth_class"].value_counts().plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Depth Class Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "validation_summary.png"), dpi=120)
    print(f"Plots saved to {PLOTS_DIR}/validation_summary.png")


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows for validation.\n")

    checks = run_structural_checks(df)
    variance_report = check_within_crop_variance(df)

    report_lines = ["=== STRUCTURAL VALIDATION ===\n"]
    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        line = f"[{status}] {name} -- {detail}"
        print(line)
        report_lines.append(line)

    report_lines.append("\n=== WITHIN-CROP DEPTH VARIANCE ===")
    report_lines.append("(low std_as_pct_of_range means soil/weather barely move the")
    report_lines.append(" needle for that crop -- depth is mostly determined by crop identity)\n")
    print("\n=== WITHIN-CROP DEPTH VARIANCE ===")
    print(variance_report.to_string(index=False))
    report_lines.append(variance_report.to_string(index=False))

    generate_plots(df)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    if all_passed:
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nAll structural checks passed. Saved validated dataset -> {OUTPUT_PATH}")
    else:
        print(f"\nSome structural checks FAILED -- fix before proceeding. See {REPORT_PATH}")

    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()