import streamlit as st
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry

BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "model" / "model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"


@st.cache_resource
def load_model():
    # pickle loads our own trained sklearn models — not untrusted external content
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


AIRPORT_COORDS = {
    "KJFK — New York JFK":   (40.639, -73.779),
    "KORD — Chicago O'Hare": (41.978, -87.904),
    "KDEN — Denver":         (39.856, -104.674),
}


def get_weather(lat: float, lon: float) -> dict:
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat, "longitude": lon,
        "current": ["temperature_2m", "wind_speed_10m", "rain", "snowfall"],
        "wind_speed_unit": "kmh"
    }
    current = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0].Current()
    return {
        "temperature": float(current.Variables(0).Value()),
        "wind_speed":  float(current.Variables(1).Value()),
        "rain":        float(current.Variables(2).Value()),
        "snow":        float(current.Variables(3).Value()),
    }


st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")
st.title("Flight Delay Predictor")
st.caption("Predicts delay probability using real-time weather + BTS-trained model")

airport_label = st.selectbox("Departure Airport", list(AIRPORT_COORDS.keys()))
hour  = st.slider("Departure Hour (UTC)", 0, 23, datetime.utcnow().hour)
month = st.number_input("Month", 1, 12, datetime.utcnow().month)

st.subheader("Known Delay Context (optional)")
col_a, col_b = st.columns(2)
weather_delay = col_a.number_input("Weather Delay (min)", 0, 500, 0)
carrier_delay = col_b.number_input("Carrier Delay (min)", 0, 500, 0)
nas_delay     = col_a.number_input("NAS Delay (min)", 0, 500, 0)
late_ac_delay = col_b.number_input("Late Aircraft Delay (min)", 0, 500, 0)

if st.button("Get Prediction"):
    lat, lon = AIRPORT_COORDS[airport_label]
    with st.spinner("Fetching live weather..."):
        try:
            w = get_weather(lat, lon)
        except Exception as e:
            st.error(f"Weather fetch failed: {e}")
            st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temp",  f"{w['temperature']:.1f} C")
    col2.metric("Wind",  f"{w['wind_speed']:.1f} km/h")
    col3.metric("Rain",  f"{w['rain']:.1f} mm/h")
    col4.metric("Snow",  f"{w['snow']:.1f} mm/h")

    model, scaler = load_model()
    features = np.array([[
        w["temperature"], w["wind_speed"], w["rain"], w["snow"],
        hour, month,
        weather_delay, carrier_delay, nas_delay, late_ac_delay
    ]])
    prob = model.predict_proba(scaler.transform(features))[0][1]

    st.subheader("Delay Prediction")
    st.metric("Probability of 15+ min delay", f"{prob * 100:.1f}%")

    if prob > 0.6:
        st.error("High delay risk")
    elif prob > 0.35:
        st.warning("Moderate delay risk")
    else:
        st.success("Low delay risk")
