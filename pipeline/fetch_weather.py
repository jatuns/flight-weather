import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime

AIRPORT_COORDS = {
    "KJFK": (40.639, -73.779),
    "KORD": (41.978, -87.904),
    "KDEN": (39.856, -104.674),
}


def wmo_code_to_description(code: int) -> str:
    mapping = {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "icy fog", 51: "light drizzle", 53: "drizzle",
        55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
        81: "heavy rain showers", 95: "thunderstorm"
    }
    return mapping.get(code, f"weather_code_{code}")


def fetch_weather(airport: str) -> dict:
    """Fetch current weather from Open-Meteo (free, no API key).

    Checks the database first to avoid redundant API calls on Render,
    where the file-based cache does not persist across restarts.
    """
    if airport not in AIRPORT_COORDS:
        raise ValueError(f"No coordinates for airport: {airport}")

    # Return cached DB row if we already have data from the last hour
    try:
        from load_postgres import get_recent_weather
        cached = get_recent_weather(airport)
        if cached is not None:
            print(f"[weather] {airport}: using cached DB row from {cached['timestamp']}")
            return cached
    except Exception as e:
        print(f"[weather] {airport}: DB cache check failed ({e}), falling back to API")

    lat, lon = AIRPORT_COORDS[airport]
    # Memory-only cache (expire_after=3600) — avoids stale file cache issues on Render
    cache_session = requests_cache.CachedSession(backend="memory", expire_after=3600)
    retry_session = retry(cache_session, retries=2, backoff_factor=0.5)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat, "longitude": lon,
        "current": ["temperature_2m", "wind_speed_10m", "rain", "snowfall", "weather_code"],
        "wind_speed_unit": "kmh"
    }

    current = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0].Current()
    code = int(current.Variables(4).Value())

    return {
        "airport": airport,
        "timestamp": datetime.utcnow().replace(minute=0, second=0, microsecond=0),
        "temperature": float(current.Variables(0).Value()),
        "wind_speed":  float(current.Variables(1).Value()),
        "visibility":  None,
        "rain":        float(current.Variables(2).Value()),
        "snow":        float(current.Variables(3).Value()),
        "weather_description": wmo_code_to_description(code)
    }
