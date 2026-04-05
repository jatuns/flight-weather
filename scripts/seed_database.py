"""
One-time script to seed the database with historical data.

Data sources:
  - BTS (Bureau of Transportation Statistics): real historical flights with actual delay data
    Download from: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ
    Required fields: FL_DATE, OP_CARRIER, ORIGIN, DEST, DEP_TIME, CRS_DEP_TIME,
                     DEP_DELAY, WEATHER_DELAY, CARRIER_DELAY, NAS_DELAY,
                     SECURITY_DELAY, LATE_AIRCRAFT_DELAY
    Save CSV as: data/bts_flights.csv

  - Open-Meteo archive: real historical hourly weather (free, no key required)

Run order:
  1. Download BTS CSV and place in data/bts_flights.csv
  2. python scripts/seed_database.py
  3. python model/train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from load_postgres import insert_weather, insert_flight
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
BTS_CSV  = DATA_DIR / "bts_flights.csv"

# BTS IATA codes → ICAO + coordinates
AIRPORTS = {
    "JFK": {"icao": "KJFK", "lat": 40.639, "lon": -73.779},
    "ORD": {"icao": "KORD", "lat": 41.978, "lon": -87.904},
    "DEN": {"icao": "KDEN", "lat": 39.856, "lon": -104.674},
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


def seed_weather(iata: str, start_date: str, end_date: str):
    """Fetch historical hourly weather from Open-Meteo archive (free, no key)."""
    info = AIRPORTS[iata]
    icao = info["icao"]

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": info["lat"],
        "longitude": info["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "wind_speed_10m", "rain", "snowfall", "weather_code"],
        "wind_speed_unit": "kmh"
    }

    responses = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    hourly = responses[0].Hourly()

    times = pd.date_range(
        start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    for i, ts in enumerate(times):
        code = int(hourly.Variables(4).ValuesAsNumpy()[i])
        insert_weather({
            "airport": icao,
            "timestamp": ts.to_pydatetime().replace(tzinfo=None),
            "temperature": float(hourly.Variables(0).ValuesAsNumpy()[i]),
            "wind_speed":  float(hourly.Variables(1).ValuesAsNumpy()[i]),
            "visibility":  None,
            "rain":        float(hourly.Variables(2).ValuesAsNumpy()[i]),
            "snow":        float(hourly.Variables(3).ValuesAsNumpy()[i]),
            "weather_description": wmo_code_to_description(code)
        })
    print(f"Weather seeded: {icao} ({iata}) {start_date} → {end_date}")


def seed_flights_from_bts():
    """
    Load real historical flight data from BTS CSV.

    Download from:
    https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ

    Required columns: FL_DATE, OP_CARRIER, ORIGIN, DEST, CRS_DEP_TIME,
                      DEP_TIME, DEP_DELAY, WEATHER_DELAY, CARRIER_DELAY,
                      NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY
    """
    if not BTS_CSV.exists():
        print(f"ERROR: {BTS_CSV} not found.")
        print("Download BTS data from:")
        print("  https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ")
        print("Save as data/bts_flights.csv and re-run.")
        return

    df = pd.read_csv(BTS_CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Filter to our airports only
    iata_codes = list(AIRPORTS.keys())
    df = df[df["ORIGIN"].isin(iata_codes)].copy()

    # Parse departure datetime
    def parse_dep_time(row):
        try:
            t = str(int(row["CRS_DEP_TIME"])).zfill(4)
            return datetime.strptime(f"{row['FL_DATE']} {t}", "%Y-%m-%d %H%M")
        except Exception:
            return None

    df["scheduled_dep"] = df.apply(parse_dep_time, axis=1)

    def parse_actual_time(row):
        try:
            t = str(int(row["DEP_TIME"])).zfill(4)
            return datetime.strptime(f"{row['FL_DATE']} {t}", "%Y-%m-%d %H%M")
        except Exception:
            return None

    df["actual_dep"] = df.apply(parse_actual_time, axis=1)

    count = 0
    for _, row in df.iterrows():
        iata = row["ORIGIN"]
        icao = AIRPORTS[iata]["icao"]

        insert_flight({
            "icao24": None,
            "date": row["scheduled_dep"],
            "actual_departure": row["actual_dep"],
            "scheduled_departure": row["scheduled_dep"],
            "departure_airport": icao,
            "arrival_airport": str(row.get("DEST", ""))[:10] if pd.notna(row.get("DEST")) else None,
            "delay_minutes": int(row["DEP_DELAY"]) if pd.notna(row.get("DEP_DELAY")) else None,
            "weather_delay_min": int(row["WEATHER_DELAY"]) if pd.notna(row.get("WEATHER_DELAY")) else None,
            "carrier_delay_min": int(row["CARRIER_DELAY"]) if pd.notna(row.get("CARRIER_DELAY")) else None,
            "nas_delay_min": int(row["NAS_DELAY"]) if pd.notna(row.get("NAS_DELAY")) else None,
            "security_delay_min": int(row["SECURITY_DELAY"]) if pd.notna(row.get("SECURITY_DELAY")) else None,
            "late_aircraft_delay_min": int(row["LATE_AIRCRAFT_DELAY"]) if pd.notna(row.get("LATE_AIRCRAFT_DELAY")) else None,
            "flight_status": "departed",
            "airline": str(row.get("OP_CARRIER", ""))[:100]
        })
        count += 1

    print(f"Flights seeded from BTS: {count} rows")


if __name__ == "__main__":
    # Determine date range from BTS CSV
    if not BTS_CSV.exists():
        seed_flights_from_bts()  # will print instructions and exit
        sys.exit(1)

    df_dates = pd.read_csv(BTS_CSV, usecols=["FL_DATE"], low_memory=False)
    start_date = pd.to_datetime(df_dates["FL_DATE"]).min().strftime("%Y-%m-%d")
    end_date   = pd.to_datetime(df_dates["FL_DATE"]).max().strftime("%Y-%m-%d")

    print(f"Seeding weather for {start_date} → {end_date}")
    for iata in AIRPORTS:
        seed_weather(iata, start_date, end_date)

    print("Seeding flights from BTS CSV...")
    seed_flights_from_bts()

    print("Database seeding complete.")
