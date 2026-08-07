"""
crop_database.py

Agronomic ground-truth data for sowing depth calculation.
Sources: ICAR crop production manuals, FAO crop guidelines, KVK
extension booklets. Depth ranges are the published safe range for
each crop under normal conditions; adjustment rules shift within 
that range based on soil and weather.
"""

CROP_DATA = {
    "Wheat": {
        "season": "Rabi",
        "seed_size_mm": 5.5,
        "seed_weight_1000g": 38,
        "base_depth_min_cm": 4.0,
        "base_depth_max_cm": 5.0,
        "optimal_depth_cm": 4.5,
    },
    "Rice": {
        "season": "Kharif",
        "seed_size_mm": 3.0,
        "seed_weight_1000g": 22,
        "base_depth_min_cm": 1.5,   
        "base_depth_max_cm": 3.0,
        "optimal_depth_cm": 2.0,
    },
    "Maize": {
        "season": "Kharif",
        "seed_size_mm": 9.0,
        "seed_weight_1000g": 280,
        "base_depth_min_cm": 4.0,
        "base_depth_max_cm": 6.0,
        "optimal_depth_cm": 5.0,
    },
    "Chickpea": {
        "season": "Rabi",
        "seed_size_mm": 8.0,
        "seed_weight_1000g": 200,   
        "base_depth_min_cm": 6.0,
        "base_depth_max_cm": 8.0,
        "optimal_depth_cm": 7.0,
    },
    "Mustard": {
        "season": "Rabi",
        "seed_size_mm": 1.8,
        "seed_weight_1000g": 4.5,
        "base_depth_min_cm": 1.5,
        "base_depth_max_cm": 2.5,
        "optimal_depth_cm": 2.0,
    },
    "Soybean": {
        "season": "Kharif",
        "seed_size_mm": 6.5,
        "seed_weight_1000g": 120,
        "base_depth_min_cm": 3.0,
        "base_depth_max_cm": 4.0,
        "optimal_depth_cm": 3.5,
    },
    "Pearl Millet": {
        "season": "Kharif",
        "seed_size_mm": 3.5,
        "seed_weight_1000g": 9,
        "base_depth_min_cm": 3.0,
        "base_depth_max_cm": 5.0,
        "optimal_depth_cm": 4.0,
    },
    "Pigeon Pea": {
        "season": "Kharif",
        "seed_size_mm": 7.0,
        "seed_weight_1000g": 90,
        "base_depth_min_cm": 4.0,
        "base_depth_max_cm": 5.0,
        "optimal_depth_cm": 4.5,
    },
}

def texture_adjustment(soil_texture_class: str) -> float:
    """Sandy = go deeper (chases moisture). Clay = go shallower (avoid crusting)."""
    rules = {
        "Sandy": +0.5,
        "Sandy Loam": +0.3,
        "Loam": 0.0,
        "Clay Loam": -0.3,
        "Clay": -0.5,
    }
    return rules.get(soil_texture_class, 0.0)


def moisture_adjustment(soil_moisture_pct: float) -> float:
    """
    Continuous version: dry topsoil -> deeper, wet topsoil -> shallower.
    Anchor points: 5% moisture -> +0.6cm, 20% -> 0cm, 35% -> -0.4cm,
    linearly interpolated between, flat beyond the ends.
    """
    m = soil_moisture_pct
    if m <= 5:
        return 0.6
    elif m <= 20:
        return 0.6 + (m - 5) * (0 - 0.6) / (20 - 5)
    elif m <= 35:
        return 0 + (m - 20) * (-0.4 - 0) / (35 - 20)
    else:
        return -0.4


def temperature_adjustment(temperature_c: float, season: str) -> float:
    """
    Continuous version. Rabi: cold mornings -> shallower (warmth-seeking).
    Any season: extreme heat -> deeper (cooler soil).
    """
    delta = 0.0
    if season == "Rabi" and temperature_c < 18:
        delta += max(-0.4, (temperature_c - 18) * (-0.4) / (5 - 18))
    if temperature_c > 32:
        delta += min(0.5, (temperature_c - 32) * 0.5 / (45 - 32))
    return round(delta, 3)


def rain_adjustment(rain_expected: bool, rainfall_7day_mm: float) -> float:
    """
    Continuous version: more expected rain -> shallower, up to a cap
    (avoid over-shallowing before very heavy rain, waterlogging risk).
    """
    if not rain_expected:
        return 0.0
    delta = -0.2 - min(0.35, (rainfall_7day_mm - 5) * 0.35 / 25)
    return round(max(delta, -0.55), 3)

def calculate_sowing_depth(
    crop_name: str,
    soil_texture_class: str,
    soil_moisture_pct: float,
    rain_expected: bool,
    rainfall_7day_mm: float,
    temperature_c: float,
) -> dict:
    """
    Returns the recommended depth plus a breakdown of why, so the
    output is explainable — not a black-box number.
    """
    if crop_name not in CROP_DATA:
        raise ValueError(f"Unknown crop: {crop_name}. Valid options: {list(CROP_DATA.keys())}")

    crop = CROP_DATA[crop_name]

    deltas = {
        "texture": texture_adjustment(soil_texture_class),
        "moisture": moisture_adjustment(soil_moisture_pct),
        "rain": rain_adjustment(rain_expected, rainfall_7day_mm),
        "temperature": temperature_adjustment(temperature_c, crop["season"]),
    }

    raw_depth = crop["optimal_depth_cm"] + sum(deltas.values())

    min_allowed = crop["base_depth_min_cm"] - 0.3
    max_allowed = crop["base_depth_max_cm"] + 0.3
    final_depth = round(min(max(raw_depth, min_allowed), max_allowed), 2)

    return {
        "crop": crop_name,
        "recommended_depth_cm": final_depth,
        "base_optimal_cm": crop["optimal_depth_cm"],
        "safe_range_cm": (crop["base_depth_min_cm"], crop["base_depth_max_cm"]),
        "adjustments_applied": deltas,
    }


if __name__ == "__main__":
    for crop_name in ["Wheat", "Chickpea", "Pearl Millet", "Pigeon Pea"]:
        result = calculate_sowing_depth(
            crop_name=crop_name,
            soil_texture_class="Sandy",
            soil_moisture_pct=10,
            rain_expected=False,
            rainfall_7day_mm=0,
            temperature_c=14,
        )
        print(result)