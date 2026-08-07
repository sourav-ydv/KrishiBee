"""
collect_weather_data.py

Pulls long-term climatological weather averages from the NASA POWER
API for the same coordinates collected in collect_soil_data.py.
For each coordinate, generates TWO rows: one for the Rabi sowing
window (~November) and one for the Kharif sowing window (~June).

NASA POWER climatology docs:
https://power.larc.nasa.gov/docs/services/api/temporal/climatology/
"""

import requests
import pandas as pd
import time
import json
import os

POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

PARAMETERS = ["T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "WS2M"]

SOWING_MONTHS = {
    "Rabi": 11,
    "Kharif": 6,
}

MONTH_KEYS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

SOIL_INPUT_PATH = "data/raw/soil_data_raw.csv"
RAW_OUTPUT_PATH = "data/raw/weather_data_raw.csv"
CHECKPOINT_PATH = "data/raw/weather_data_checkpoint.json"


def fetch_climatology_point(lat: float, lon: float, retries: int = 3) -> dict:
    """Fetch monthly climatology for one coordinate. Returns raw JSON properties, or None."""
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(POWER_URL, params=params, timeout=25)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print("  Rate limited, waiting 10s...")
                time.sleep(10)
            else:
                print(f"  HTTP {resp.status_code} for ({lat}, {lon})")
        except requests.exceptions.RequestException as e:
            print(f"  Request failed for ({lat}, {lon}): {e}")
        time.sleep(2)
    return None


def estimate_rain_proxy(monthly_precip_mm_day: float, all_months_precip: list) -> tuple:
    """
    Derive a proxy rain_expected flag + estimated 7-day rainfall from
    the monthly climatological daily average. A month is flagged
    'rain expected' if its avg daily precip is above the location's
    own median month (i.e. relatively wet for that specific place).
    """
    rainfall_7day_mm = round(monthly_precip_mm_day * 7, 2)
    median_precip = sorted(all_months_precip)[len(all_months_precip) // 2]
    rain_expected = monthly_precip_mm_day > median_precip
    return rainfall_7day_mm, rain_expected


def parse_season_rows(data: dict, lat: float, lon: float, state: str) -> list:
    """Extract Rabi and Kharif rows from one point's climatology response."""
    try:
        params_data = data["properties"]["parameter"]
    except (KeyError, TypeError):
        print(f"  Failed to parse response for ({lat}, {lon})")
        return []

    rows = []
    all_precip_months = [params_data["PRECTOTCORR"][m] for m in MONTH_KEYS]

    for season, month_num in SOWING_MONTHS.items():
        month_key = MONTH_KEYS[month_num - 1]
        try:
            temp = params_data["T2M"][month_key]
            humidity = params_data["RH2M"][month_key]
            precip_day = params_data["PRECTOTCORR"][month_key]
            solar = params_data["ALLSKY_SFC_SW_DWN"][month_key]
            wind = params_data["WS2M"][month_key]
        except KeyError:
            continue

        rainfall_7day_mm, rain_expected = estimate_rain_proxy(precip_day, all_precip_months)

        rows.append({
            "latitude": lat,
            "longitude": lon,
            "state": state,
            "season": season,
            "sowing_month": month_num,
            "temperature_C": round(temp, 2),
            "humidity_pct": round(humidity, 2),
            "rainfall_7day_mm": rainfall_7day_mm,
            "rain_expected": rain_expected,
            "solar_radiation_MJ": round(solar, 2),
            "wind_speed_ms": round(wind, 2),
        })

    return rows


def load_checkpoint() -> list:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return []


def save_checkpoint(results: list):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f)


def main():
    if not os.path.exists(SOIL_INPUT_PATH):
        print(f"ERROR: {SOIL_INPUT_PATH} not found. Run collect_soil_data.py first.")
        return

    soil_df = pd.read_csv(SOIL_INPUT_PATH)
    coordinates = soil_df[["latitude", "longitude", "state"]].to_dict("records")
    print(f"Loaded {len(coordinates)} coordinates from soil data.")

    results = load_checkpoint()
    done_coords = {(r["latitude"], r["longitude"]) for r in results}
    print(f"Resuming: {len(done_coords)} coordinates already collected.")

    for i, coord in enumerate(coordinates):
        lat, lon, state = coord["latitude"], coord["longitude"], coord["state"]
        if (lat, lon) in done_coords:
            continue

        print(f"[{i+1}/{len(coordinates)}] Fetching ({state}) {lat}, {lon}...")
        data = fetch_climatology_point(lat, lon)
        if data:
            season_rows = parse_season_rows(data, lat, lon, state)
            results.extend(season_rows)
            done_coords.add((lat, lon))

        if (i + 1) % 10 == 0:
            save_checkpoint(results)
            print(f"  Checkpoint saved: {len(results)} season-rows so far.")

        time.sleep(1.0)

    save_checkpoint(results)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)
    df.to_csv(RAW_OUTPUT_PATH, index=False)
    print(f"\nDone. Saved {len(df)} rows to {RAW_OUTPUT_PATH}")
    print(f"(that's {len(df)} season-rows from {len(done_coords)} unique coordinates)")
    print(df.head())


if __name__ == "__main__":
    main()