"""
One-time script to seed the database with historical data.

Data sources:
  - 2015 Flight Delays dataset (Kaggle/BTS):
    Place the flights.csv file at: data/archive/flights.csv

  - Open-Meteo archive: real historical hourly weather (free, no key required)

Run order:
  1. Place data/archive/flights.csv in the data directory
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
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path
from load_postgres import insert_weather, get_conn
from dotenv import load_dotenv

load_dotenv()

DATA_DIR    = Path(__file__).parent.parent / "data"
FLIGHTS_CSV = DATA_DIR / "archive" / "flights.csv"

# IATA codes → ICAO + coordinates
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


def seed_flights_from_csv():
    """
    Load real historical flight data from 2015 Flight Delays dataset.
    Expected at: data/archive/flights.csv

    Columns used: YEAR, MONTH, DAY, AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT,
                  SCHEDULED_DEPARTURE, DEPARTURE_TIME, DEPARTURE_DELAY,
                  WEATHER_DELAY, AIRLINE_DELAY, AIR_SYSTEM_DELAY,
                  SECURITY_DELAY, LATE_AIRCRAFT_DELAY
    """
    if not FLIGHTS_CSV.exists():
        print(f"ERROR: {FLIGHTS_CSV} not found.")
        return

    iata_codes = list(AIRPORTS.keys())

    # Read only needed columns, filter to our airports
    cols = [
        "YEAR", "MONTH", "DAY", "AIRLINE",
        "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY",
        "WEATHER_DELAY", "AIRLINE_DELAY", "AIR_SYSTEM_DELAY",
        "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
    ]
    df = pd.read_csv(FLIGHTS_CSV, usecols=cols, low_memory=False)
    df = df[df["ORIGIN_AIRPORT"].isin(iata_codes)].copy()
    print(f"Loaded {len(df)} flights for {iata_codes}")

    def parse_datetime(row):
        try:
            t = str(int(row["SCHEDULED_DEPARTURE"])).zfill(4)
            return datetime(int(row["YEAR"]), int(row["MONTH"]), int(row["DAY"]),
                            int(t[:2]), int(t[2:]))
        except Exception:
            return None

    def parse_actual(row):
        try:
            t = str(int(row["DEPARTURE_TIME"])).zfill(4)
            return datetime(int(row["YEAR"]), int(row["MONTH"]), int(row["DAY"]),
                            int(t[:2]), int(t[2:]))
        except Exception:
            return None

    df["scheduled_dep"] = df.apply(parse_datetime, axis=1)
    df["actual_dep"]    = df.apply(parse_actual, axis=1)

    from uuid import uuid4

    def to_int(val):
        try:
            return int(val) if pd.notna(val) else None
        except Exception:
            return None

    rows = []
    for _, row in df.iterrows():
        iata = row["ORIGIN_AIRPORT"]
        icao = AIRPORTS[iata]["icao"]
        scheduled = row["scheduled_dep"] if pd.notna(row["scheduled_dep"]) else None
        actual    = row["actual_dep"]    if pd.notna(row["actual_dep"])    else None
        rows.append((
            str(uuid4()),
            None,
            scheduled,
            icao,
            str(row["DESTINATION_AIRPORT"])[:10] if pd.notna(row.get("DESTINATION_AIRPORT")) else None,
            scheduled,
            actual,
            to_int(row.get("DEPARTURE_DELAY")),
            to_int(row.get("WEATHER_DELAY")),
            to_int(row.get("AIRLINE_DELAY")),
            to_int(row.get("AIR_SYSTEM_DELAY")),
            to_int(row.get("SECURITY_DELAY")),
            to_int(row.get("LATE_AIRCRAFT_DELAY")),
            "departed",
            str(row.get("AIRLINE", ""))[:100]
        ))

    BATCH = 1000
    conn = get_conn()
    cur  = conn.cursor()
    inserted = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        cur.executemany("""
            INSERT INTO flights (
                flight_id, icao24, date, departure_airport, arrival_airport,
                scheduled_departure, actual_departure, delay_minutes,
                weather_delay_min, carrier_delay_min, nas_delay_min,
                security_delay_min, late_aircraft_delay_min,
                flight_status, airline
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (flight_id) DO NOTHING
        """, batch)
        conn.commit()
        inserted += len(batch)
        print(f"  Inserted {inserted}/{len(rows)} flights...", end="\r")
    cur.close()
    conn.close()

    print(f"\nFlights seeded: {inserted} rows")


if __name__ == "__main__":
    if not FLIGHTS_CSV.exists():
        print(f"ERROR: {FLIGHTS_CSV} not found.")
        print("Place the flights.csv file at data/archive/flights.csv and re-run.")
        sys.exit(1)

    # 2015 dataset covers full year 2015
    start_date = "2015-01-01"
    end_date   = "2015-12-31"

    print(f"Seeding weather for {start_date} → {end_date}")
    for iata in AIRPORTS:
        seed_weather(iata, start_date, end_date)

    print("Seeding flights from CSV...")
    seed_flights_from_csv()

    print("Database seeding complete.")
