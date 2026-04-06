import psycopg2
import os
from uuid import uuid4


def get_conn():
    host = os.getenv("DB_HOST", "localhost")
    sslmode = "require" if host != "localhost" else "prefer"
    return psycopg2.connect(
        host=host,
        dbname=os.getenv("DB_NAME", "flight_weather"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        sslmode=sslmode
    )


def insert_weather(weather: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO weather (airport, timestamp, temperature, wind_speed,
                             visibility, rain, snow, weather_description)
        VALUES (%(airport)s, %(timestamp)s, %(temperature)s, %(wind_speed)s,
                %(visibility)s, %(rain)s, %(snow)s, %(weather_description)s)
        ON CONFLICT (airport, timestamp) DO NOTHING
    """, weather)
    conn.commit()
    cur.close()
    conn.close()


def insert_flight(flight: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO flights (
            flight_id, icao24, date, departure_airport, arrival_airport,
            scheduled_departure, actual_departure, delay_minutes,
            weather_delay_min, carrier_delay_min, nas_delay_min,
            security_delay_min, late_aircraft_delay_min,
            flight_status, airline
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (flight_id) DO NOTHING
    """, (
        str(uuid4()),
        flight.get("icao24"),
        flight.get("date"),
        flight.get("departure_airport"),
        flight.get("arrival_airport"),
        flight.get("scheduled_departure"),
        flight.get("actual_departure"),
        flight.get("delay_minutes"),
        flight.get("weather_delay_min"),
        flight.get("carrier_delay_min"),
        flight.get("nas_delay_min"),
        flight.get("security_delay_min"),
        flight.get("late_aircraft_delay_min"),
        flight.get("flight_status", "departed"),
        flight.get("airline")
    ))
    conn.commit()
    cur.close()
    conn.close()
