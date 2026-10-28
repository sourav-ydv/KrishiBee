import requests
from datetime import datetime
from fastapi import APIRouter, HTTPException

router = APIRouter()

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def sky_label(code):
    if code is None:
        return "Unknown"
    code = int(code)
    if code == 0:
        return "Clear"
    if code in (1, 2, 3):
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57):
        return "Drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "Rain"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snow"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return "Mixed"


def short_date(iso_date):
    return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d %b")


@router.get("/forecast")
def forecast(latitude: float, longitude: float):
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise HTTPException(400, "That point is outside the map.")

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
                "forecast_days": 7,
                "timezone": "auto",
            },
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        raise HTTPException(502, "The 7-day forecast is unavailable right now. Try again shortly.")

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    if len(dates) < 7:
        raise HTTPException(502, "The forecast came back incomplete. Try again shortly.")

    days = []
    for i, iso_date in enumerate(dates[:7]):
        rain_mm = daily["precipitation_sum"][i]
        probability = daily["precipitation_probability_max"][i]
        days.append({
            "date": iso_date,
            "label": short_date(iso_date),
            "sky": sky_label(daily["weather_code"][i]),
            "temp_min_c": None if daily["temperature_2m_min"][i] is None else round(daily["temperature_2m_min"][i], 1),
            "temp_max_c": None if daily["temperature_2m_max"][i] is None else round(daily["temperature_2m_max"][i], 1),
            "rain_mm": None if rain_mm is None else round(rain_mm, 1),
            "rain_probability_pct": None if probability is None else int(round(probability)),
        })

    rainy = [
        day for day in days
        if (day["rain_mm"] or 0) >= 1 or (day["rain_probability_pct"] or 0) >= 50
    ]
    if rainy:
        rain_line = "Rain is likely on " + ", ".join(day["label"] for day in rainy) + "."
    else:
        rain_line = "Little or no rain is forecast for the next 7 days."

    return {
        "latitude": payload.get("latitude", latitude),
        "longitude": payload.get("longitude", longitude),
        "timezone": payload.get("timezone"),
        "days": days,
        "rain_likely": bool(rainy),
        "rain_line": rain_line,
        "source": "Open-Meteo",
        "note": "This is a weather forecast, not a PAU recommendation. Check the field before you irrigate.",
    }