"""
FastAPI backend for Flight Delay Predictor.
Run from project root: uvicorn fastapi_app.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import pickle  # loading our own trained sklearn/xgboost model — not untrusted content
import numpy as np
from datetime import datetime
import openmeteo_requests
import requests_cache
from retry_requests import retry

BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "model" / "model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
HTML_PATH   = Path(__file__).parent / "index.html"

AIRPORTS = {
    "JFK": {"lat": 40.639, "lon": -73.779, "icao": "KJFK"},
    "ORD": {"lat": 41.978, "lon": -87.904, "icao": "KORD"},
    "DEN": {"lat": 39.856, "lon": -104.674, "icao": "KDEN"},
}

FORECAST_DAYS = 16

try:
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found. Run model/train.py first.\n{e}")

app = FastAPI(title="Flight Delay Predictor")


class PredictRequest(BaseModel):
    airport: str
    date: str
    hour: int
    weather_delay: int = 0
    carrier_delay: int = 0
    nas_delay: int = 0
    late_aircraft: int = 0


def get_forecast_weather(lat: float, lon: float, target_dt: datetime) -> dict:
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "wind_speed_10m", "rain", "snowfall"],
        "wind_speed_unit": "kmh",
        "forecast_days": FORECAST_DAYS,
    }
    response = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
    hourly = response.Hourly()
    n = hourly.Variables(0).ValuesAsNumpy().shape[0]
    times = [datetime.utcfromtimestamp(hourly.Time() + i * hourly.Interval()) for i in range(n)]
    target_naive = target_dt.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    idx = min(range(len(times)), key=lambda i: abs((times[i] - target_naive).total_seconds()))
    return {
        "temperature": float(hourly.Variables(0).ValuesAsNumpy()[idx]),
        "wind_speed":  float(hourly.Variables(1).ValuesAsNumpy()[idx]),
        "rain":        float(hourly.Variables(2).ValuesAsNumpy()[idx]),
        "snow":        float(hourly.Variables(3).ValuesAsNumpy()[idx]),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PATH.read_text(encoding="utf-8")


@app.post("/predict")
def predict(req: PredictRequest):
    if req.airport not in AIRPORTS:
        raise HTTPException(status_code=400, detail="Invalid airport")
    if not (0 <= req.hour <= 23):
        raise HTTPException(status_code=400, detail="Hour must be 0-23")
    info = AIRPORTS[req.airport]
    try:
        year, month, day = map(int, req.date.split("-"))
        target_dt = datetime(year, month, day, req.hour)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")
    try:
        w = get_forecast_weather(info["lat"], info["lon"], target_dt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {e}")
    features = np.array([[
        w["temperature"], w["wind_speed"], w["rain"], w["snow"],
        req.hour, month,
        req.weather_delay, req.carrier_delay, req.nas_delay, req.late_aircraft,
    ]])
    prob = float(_model.predict_proba(_scaler.transform(features))[0][1])
    return {
        "probability": round(prob, 4),
        "temperature": round(w["temperature"], 1),
        "wind_speed":  round(w["wind_speed"], 1),
        "rain":        round(w["rain"], 2),
        "snow":        round(w["snow"], 2),
        "icao":        info["icao"],
    }
