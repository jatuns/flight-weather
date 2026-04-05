import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from pathlib import Path
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}/{os.getenv('DB_NAME', 'flight_weather')}"
)

QUERY = """
SELECT
    f.delay_minutes,
    f.weather_delay_min,
    f.carrier_delay_min,
    f.nas_delay_min,
    f.late_aircraft_delay_min,
    EXTRACT(HOUR  FROM f.date) AS hour_of_day,
    EXTRACT(MONTH FROM f.date) AS month,
    w.temperature,
    w.wind_speed,
    w.rain,
    w.snow
FROM flights f
LEFT JOIN weather w
    ON f.departure_airport = w.airport
    AND DATE_TRUNC('hour', f.date) = DATE_TRUNC('hour', w.timestamp)
WHERE f.delay_minutes IS NOT NULL
  AND w.wind_speed    IS NOT NULL
  AND w.temperature   IS NOT NULL
"""

FEATURES = [
    "temperature", "wind_speed", "rain", "snow",
    "hour_of_day", "month",
    "weather_delay_min", "carrier_delay_min",
    "nas_delay_min", "late_aircraft_delay_min"
]

MODEL_DIR = Path(__file__).parent


def train():
    engine = create_engine(DB_URL)
    df = pd.read_sql(QUERY, engine)

    if len(df) < 200:
        raise ValueError(
            f"Only {len(df)} usable rows. Need 200+. Run scripts/seed_database.py first."
        )

    df["delayed"] = (df["delay_minutes"] > 15).astype(int)

    # Fill NULL delay components with 0 (no delay of that type)
    delay_cols = ["weather_delay_min", "carrier_delay_min", "nas_delay_min", "late_aircraft_delay_min"]
    df[delay_cols] = df[delay_cols].fillna(0)

    df = df[FEATURES + ["delayed"]].dropna()

    print(f"Training on {len(df)} rows | Delay rate: {df['delayed'].mean()*100:.1f}%")

    X, y = df[FEATURES], df["delayed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Baseline (always on-time)": None,
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            eval_metric="logloss"
        ),
    }

    best_auc = 0
    best_model = None
    best_name = ""

    for name, m in models.items():
        if m is None:
            # Baseline: predict majority class (on-time)
            y_pred = [0] * len(y_test)
            y_prob = [0.0] * len(y_test)
        else:
            m.fit(X_train_s, y_train)
            y_pred = m.predict(X_test_s)
            y_prob = m.predict_proba(X_test_s)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {auc:.4f}")

        if m is not None and auc > best_auc:
            best_auc = auc
            best_model = m
            best_name = name

    print(f"\nBest model: {best_name} (AUC: {best_auc:.4f})")

    # Serialize our own trained sklearn/xgboost models — not loading untrusted content
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Best model and scaler saved.")


if __name__ == "__main__":
    train()
