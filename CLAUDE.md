# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Flight Weather Project — Claude Context

## Commands

```bash
# Seed database (run once, from repo root)
python scripts/seed_database.py

# Train / retrain ML models
python model/train.py

# Run Streamlit app locally
streamlit run streamlit_app/app.py

# Start live pipeline (Prefect, every 6 hours) — must run from pipeline/ dir
cd pipeline && python prefect_flow.py
```

> **Note:** `prefect_flow.py` uses bare module imports (`from fetch_flights import ...`), so it must be invoked from the `pipeline/` directory.

## Project Goal
End-to-end data analytics portfolio project. Collects real flight + weather data, stores in PostgreSQL, analyzes delay patterns, trains and compares ML models (Baseline, Logistic Regression, Random Forest, XGBoost), and serves predictions via Streamlit.

## Stack
- **Flight data (historical seed)**: BTS — Bureau of Transportation Statistics (free CSV download, real delay data)
- **Flight data (live pipeline)**: OpenSky Network API (free, no key, actual departure times)
- **Weather data**: Open-Meteo API (free, no key — current forecast + historical archive)
- **Database**: PostgreSQL (local)
- **Orchestration**: Prefect (6-hour schedule, 3x retry logic)
- **ML**: scikit-learn + XGBoost — binary classification, multi-model comparison
- **Dashboard**: Tableau (connects directly to PostgreSQL)
- **Web app**: Streamlit → deployed to Streamlit Cloud
- **Domain**: baristunatugrul.com/flight-project via Vercel redirect

## Project Structure
```
flight-weather-project/
├── data/
│   └── archive/
│       └── flights.csv          # Kaggle 2015 Flight Delays dataset, gitignored
├── pipeline/
│   ├── fetch_flights.py
│   ├── fetch_weather.py
│   ├── load_postgres.py
│   └── prefect_flow.py
├── scripts/
│   └── seed_database.py
├── sql/
│   ├── schema.sql
│   └── analysis.sql
├── analysis/
│   └── eda.ipynb
├── model/
│   ├── train.py
│   ├── model.pkl
│   └── scaler.pkl
├── streamlit_app/
│   └── app.py
├── requirements.txt
└── .env.example
```

## Database
Two tables: `flights` and `weather`
Join: `LEFT JOIN ON airport + DATE_TRUNC('hour', timestamp)`
Indexes: `idx_weather_airport_ts`, `idx_flights_airport_date`
Unique constraint: `uq_weather_airport_ts` on `(airport, timestamp)`

## Known Intentional Decisions
- `visibility` field is NULL everywhere — Open-Meteo does not provide it, excluded from ML features
- `delay_minutes` is NULL for live pipeline flights — OpenSky has no scheduled times; BTS seed data has real delays
- AviationStack removed from project — 100 call/month limit insufficient; replaced by BTS CSV
- `LEFT JOIN` used (not INNER JOIN) to preserve all flights when weather data is missing
- `class_weight="balanced"` in LogisticRegression, `scale_pos_weight` in XGBoost — dataset is imbalanced
- ML features: temperature, wind_speed, rain, snow, hour_of_day, month, weather_delay_min, carrier_delay_min, nas_delay_min, late_aircraft_delay_min
- Best model (by ROC-AUC) is auto-selected and saved as model.pkl after each train run

## Airports in Scope (US — BTS data)
| IATA | ICAO | City | Lat | Lon |
|------|------|------|-----|-----|
| JFK  | KJFK | New York | 40.639 | -73.779 |
| ORD  | KORD | Chicago | 41.978 | -87.904 |
| DEN  | KDEN | Denver | 39.856 | -104.674 |

## Seed Flight Data
`scripts/seed_database.py` uses the **Kaggle 2015 Flight Delays dataset** (not transtats BTS download).
Expected path: `data/archive/flights.csv` (gitignored due to size).
Columns consumed: `YEAR, MONTH, DAY, AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT,
SCHEDULED_DEPARTURE, DEPARTURE_TIME, DEPARTURE_DELAY, WEATHER_DELAY, AIRLINE_DELAY,
AIR_SYSTEM_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY`

Column mapping into DB: `AIRLINE_DELAY` → `carrier_delay_min`, `AIR_SYSTEM_DELAY` → `nas_delay_min`

## Critical Rules
- Never call AviationStack — removed from project
- Always use `pathlib.Path(__file__).parent` for file paths (Streamlit Cloud compatibility)
- Always use `@st.cache_resource` for model loading in Streamlit
- seed_database.py must be run before train.py (model needs 200+ rows)
- Use `ON CONFLICT DO NOTHING` on flights insert, `ON CONFLICT` on weather requires unique constraint
- Never hardcode DB credentials — always read from environment variables via `os.getenv()`
- ICAO codes used in DB and pipeline (KJFK, KORD, KDEN); IATA codes (JFK, ORD, DEN) used in BTS CSV only

## Environment Variables
```
DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
PREFECT_API_KEY, PREFECT_API_URL  (optional, only for Prefect Cloud)
```

## Setup Order
1. Create DB + run schema.sql
2. Fill .env
3. Download Kaggle 2015 Flight Delays CSV → save as data/archive/flights.csv
4. Run scripts/seed_database.py
5. Run model/train.py
6. Test streamlit_app/app.py locally
7. Start pipeline/prefect_flow.py

## ML Model
- Target: `delay_minutes > 15` → binary (0/1)
- Models compared: Baseline, Logistic Regression, Random Forest, XGBoost
- Best model selected by ROC-AUC on test set
- Min rows to train: 200
- Saved to: `model/model.pkl`, `model/scaler.pkl`
- Retrain periodically as data accumulates

## What delay_minutes NULL means
Live pipeline flights have NULL delay because OpenSky provides no scheduled time.
These rows are excluded from ML training and SQL analysis via `WHERE delay_minutes IS NOT NULL`.
They are still stored for future enrichment.
