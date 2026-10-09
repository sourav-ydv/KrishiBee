import requests
import pandas as pd
import time
import json
import os
import math

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

PROPERTIES = ["clay", "sand", "silt", "phh2o", "soc", "bdod", "nitrogen"]
DEPTH = "0-5cm"   

STATE_BOUNDS = {
    "Punjab":         (29.5, 32.5, 73.8, 76.9),
    "Haryana":        (27.6, 30.9, 74.4, 77.6),
    "Uttar Pradesh":  (23.8, 30.4, 77.0, 84.6),
    "Madhya Pradesh": (21.0, 26.9, 74.0, 82.8),
    "Rajasthan":      (23.0, 30.2, 69.5, 78.3),
}

POINTS_PER_STATE = 30 
RAW_OUTPUT_PATH = "data/raw/soil_data_raw.csv"
CHECKPOINT_PATH = "data/raw/soil_data_checkpoint.json"


def generate_grid_coordinates(bounds: tuple, n_points: int) -> list:
    min_lat, max_lat, min_lon, max_lon = bounds
    side = math.ceil(math.sqrt(n_points))
    lats = [min_lat + (max_lat - min_lat) * i / (side - 1) for i in range(side)]
    lons = [min_lon + (max_lon - min_lon) * j / (side - 1) for j in range(side)]
    points = [(round(lat, 4), round(lon, 4)) for lat in lats for lon in lons]
    return points[:n_points]


def fetch_soil_point(lat: float, lon: float, retries: int = 3, timeout: int = 20) -> dict:
    params = {"lon": lon, "lat": lat, "property": PROPERTIES, "depth": DEPTH, "value": "mean"}
    for attempt in range(retries):
        try:
            resp = requests.get(SOILGRIDS_URL, params=params, timeout=timeout)
            if resp.status_code == 200:
                return parse_soilgrids_response(resp.json(), lat, lon)
            elif resp.status_code == 429:
                print("  Rate limited, waiting 10s...")
                time.sleep(10)
            else:
                print(f"  HTTP {resp.status_code} for ({lat}, {lon})")
        except requests.exceptions.RequestException as e:
            print(f"  Request failed for ({lat}, {lon}): {e}")
        time.sleep(2)
    return None


def parse_soilgrids_response(data: dict, lat: float, lon: float) -> dict:
    row = {"latitude": lat, "longitude": lon}
    try:
        for layer in data["properties"]["layers"]:
            prop_name = layer["name"]
            for d in layer["depths"]:
                if d["label"] == DEPTH:
                    mean_val = d["values"]["mean"]
                    d_factor = layer.get("unit_measure", {}).get("d_factor", 1)
                    row[prop_name] = mean_val / d_factor if mean_val is not None else None
    except (KeyError, TypeError) as e:
        print(f"  Failed to parse response for ({lat}, {lon}): {e}")
        return None
    return row


def load_checkpoint() -> list:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return []


def save_checkpoint(results: list):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f)


def main():
    all_coordinates = []
    for state, bounds in STATE_BOUNDS.items():
        for lat, lon in generate_grid_coordinates(bounds, POINTS_PER_STATE):
            all_coordinates.append({"state": state, "latitude": lat, "longitude": lon})

    print(f"Generated {len(all_coordinates)} coordinates across {len(STATE_BOUNDS)} states.")

    results = load_checkpoint()
    done_coords = {(r["latitude"], r["longitude"]) for r in results}
    print(f"Resuming: {len(results)} already collected.")

    for i, coord in enumerate(all_coordinates):
        lat, lon, state = coord["latitude"], coord["longitude"], coord["state"]
        if (lat, lon) in done_coords:
            continue

        print(f"[{i+1}/{len(all_coordinates)}] Fetching ({state}) {lat}, {lon}...")
        row = fetch_soil_point(lat, lon)
        if row:
            row["state"] = state
            results.append(row)
            done_coords.add((lat, lon))

        if (i + 1) % 10 == 0:
            save_checkpoint(results)
            print(f"  Checkpoint saved: {len(results)} points so far.")

        time.sleep(1.5) 

    save_checkpoint(results)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)
    df.to_csv(RAW_OUTPUT_PATH, index=False)
    print(f"\nDone. Saved {len(df)} rows to {RAW_OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()