# Flight Delay Predictor

An end-to-end data engineering and machine learning project that collects real flight and weather data, stores it in PostgreSQL, trains a multi-model ML pipeline, and serves predictions through a FastAPI web application with a custom cockpit-style UI.

---

## Overview

This project answers a practical question: **given a departure airport, date, time, and current weather forecast — how likely is a 15+ minute delay?**

It is built entirely on free, public data sources with no paid API keys required, and covers the full data lifecycle from ingestion to prediction.

---

## Stack

| Layer | Technology |
|---|---|
| Historical flight data | [BTS — Bureau of Transportation Statistics](https://www.transtats.bts.gov/) |
| Live flight pipeline | [OpenSky Network API](https://opensky-network.org/) (no key required) |
| Weather data | [Open-Meteo API](https://open-meteo.com/) — forecast + historical archive (no key required) |
| Database | PostgreSQL |
| Orchestration | Prefect (6-hour schedule, 3× retry logic) |
| ML | scikit-learn · XGBoost — binary classification, multi-model comparison |
| Web app | FastAPI + plain HTML/CSS/JS |
| Dashboard | Tableau (direct PostgreSQL connection) |

---

## Project Structure

```
flight-weather/
├── fastapi_app/
│   ├── main.py          # FastAPI backend — /predict endpoint
│   └── index.html       # Frontend — cockpit UI
├── pipeline/
│   ├── fetch_flights.py # OpenSky live flight ingestion
│   ├── fetch_weather.py # Open-Meteo weather ingestion
│   ├── load_postgres.py # DB helpers (insert_weather, get_conn)
│   └── prefect_flow.py  # Orchestrated 6-hour pipeline
├── scripts/
│   └── seed_database.py # One-time historical data seeder
├── sql/
│   ├── schema.sql       # Table definitions + indexes
│   └── analysis.sql     # Exploratory SQL queries
├── analysis/
│   └── eda.ipynb        # Exploratory data analysis notebook
├── model/
│   ├── train.py         # Model training + comparison script
│   ├── model.pkl        # Best model (auto-selected, gitignored)
│   └── scaler.pkl       # Feature scaler (gitignored)
├── streamlit_app/
│   └── app.py           # Streamlit version (legacy)
├── .env.example         # Environment variable template
└── requirements.txt
```

---

## Airports in Scope

| IATA | ICAO | Airport | City |
|------|------|---------|------|
| JFK | KJFK | John F. Kennedy International | New York |
| ORD | KORD | O'Hare International | Chicago |
| DEN | KDEN | Denver International | Denver |

---

## Setup

### 1. Prerequisites

- Python 3.10+
- PostgreSQL running locally
- Virtual environment (recommended)

```bash
git clone https://github.com/jatuns/flight-weather.git
cd flight-weather
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy the example file and fill in your PostgreSQL credentials:

```bash
cp .env.example .env
```

```env
DB_HOST=localhost
DB_NAME=flight_weather
DB_USER=postgres
DB_PASSWORD=your_password

# Optional — only needed for Prefect Cloud deployment
PREFECT_API_KEY=
PREFECT_API_URL=
```

### 3. Create the database

```bash
psql -U postgres -c "CREATE DATABASE flight_weather;"
psql -U postgres -d flight_weather -f sql/schema.sql
```

### 4. Download historical flight data (BTS)

Go to the [BTS On-Time Performance page](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ), select the following fields, and download as CSV:

```
FL_DATE, OP_CARRIER, ORIGIN, DEST,
CRS_DEP_TIME, DEP_TIME, DEP_DELAY,
WEATHER_DELAY, CARRIER_DELAY, NAS_DELAY,
SECURITY_DELAY, LATE_AIRCRAFT_DELAY
```

Save the file to:

```
data/bts_flights.csv
```

> The file is gitignored because of its size (~250MB for a full year).

### 5. Seed the database

This script loads the BTS CSV into PostgreSQL and fetches historical weather from the Open-Meteo archive for the same period. It runs once.

```bash
python scripts/seed_database.py
```

Expect this to take a few minutes. It processes flights in 1,000-row batches.

### 6. Train the model

Trains and compares four models — Baseline, Logistic Regression, Random Forest, and XGBoost. The best model by ROC-AUC is saved automatically.

```bash
python model/train.py
```

Output example:
```
Baseline        ROC-AUC: 0.500
LogisticReg     ROC-AUC: 0.701
RandomForest    ROC-AUC: 0.812
XGBoost         ROC-AUC: 0.834  ← best
Model saved to model/model.pkl
```

### 7. Run the web app

```bash
uvicorn fastapi_app.main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Web App

The prediction interface is built with FastAPI serving a single-page HTML application. No JavaScript framework — just vanilla JS, CSS variables, and SVG.

**How it works:**

1. Select a departure airport (JFK / ORD / DEN)
2. Pick a date within the next 16 days and a UTC departure hour
3. Optionally enter any delay context already visible in your airline app
4. Click **Run Delay Analysis**

The app fetches the hourly weather forecast from Open-Meteo for the selected airport and time, runs the XGBoost model, and returns a delay probability with a full breakdown of the input features.

**API endpoint:**

```
POST /predict
```

```json
{
  "airport": "JFK",
  "date": "2026-04-20",
  "hour": 14,
  "weather_delay": 0,
  "carrier_delay": 0,
  "nas_delay": 0,
  "late_aircraft": 0
}
```

```json
{
  "probability": 0.3142,
  "temperature": 12.4,
  "wind_speed": 28.1,
  "rain": 0.0,
  "snow": 0.0,
  "icao": "KJFK"
}
```

---

## ML Model

| | |
|---|---|
| **Target** | `delay_minutes > 15` → binary (0 / 1) |
| **Features** | temperature, wind_speed, rain, snow, hour_of_day, month, weather_delay_min, carrier_delay_min, nas_delay_min, late_aircraft_delay_min |
| **Class imbalance** | Handled via `class_weight="balanced"` (Logistic Regression) and `scale_pos_weight` (XGBoost) |
| **Model selection** | Highest ROC-AUC on held-out test set |
| **Training data** | 2015 BTS flight records for JFK, ORD, DEN (~450,000 rows) |
| **Minimum rows** | 200 (enforced in train.py) |

---

## Live Pipeline

Once the database is seeded and the model is trained, the Prefect pipeline keeps data fresh by running every 6 hours:

```bash
python pipeline/prefect_flow.py
```

It fetches new flight states from OpenSky and current weather from Open-Meteo, then inserts into PostgreSQL. Live flights have `delay_minutes = NULL` (OpenSky provides no scheduled times) and are excluded from ML training but stored for future enrichment.

---

## Database Schema

```sql
-- Flights
CREATE TABLE flights (
    flight_id               UUID PRIMARY KEY,
    icao24                  VARCHAR(20),
    date                    TIMESTAMP,
    departure_airport       VARCHAR(10),   -- ICAO code (e.g. KJFK)
    arrival_airport         VARCHAR(10),
    scheduled_departure     TIMESTAMP,
    actual_departure        TIMESTAMP,
    delay_minutes           INTEGER,       -- NULL for live pipeline rows
    weather_delay_min       INTEGER,
    carrier_delay_min       INTEGER,
    nas_delay_min           INTEGER,
    security_delay_min      INTEGER,
    late_aircraft_delay_min INTEGER,
    flight_status           VARCHAR(20),
    airline                 VARCHAR(100)
);

-- Weather
CREATE TABLE weather (
    id                  SERIAL PRIMARY KEY,
    airport             VARCHAR(10),       -- ICAO code
    timestamp           TIMESTAMP,
    temperature         FLOAT,
    wind_speed          FLOAT,
    rain                FLOAT,
    snow                FLOAT,
    weather_description VARCHAR(100),
    CONSTRAINT uq_weather_airport_ts UNIQUE (airport, timestamp)
);
```

Tables are joined on `airport` + `DATE_TRUNC('hour', timestamp)`.

---

## Notes

- `visibility` is `NULL` everywhere — Open-Meteo does not provide it and it is excluded from ML features
- ICAO codes (`KJFK`, `KORD`, `KDEN`) are used throughout the database and pipeline; IATA codes (`JFK`, `ORD`, `DEN`) appear only in the BTS CSV
- The `.env` file is gitignored — never commit credentials

---

## License

MIT
