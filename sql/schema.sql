CREATE TABLE IF NOT EXISTS flights (
    flight_id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    icao24                  VARCHAR(20),
    date                    TIMESTAMP,
    departure_airport       VARCHAR(10),
    arrival_airport         VARCHAR(10),
    scheduled_departure     TIMESTAMP,
    actual_departure        TIMESTAMP,
    delay_minutes           INTEGER,
    weather_delay_min       INTEGER,
    carrier_delay_min       INTEGER,
    nas_delay_min           INTEGER,
    security_delay_min      INTEGER,
    late_aircraft_delay_min INTEGER,
    flight_status           VARCHAR(20),
    airline                 VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS weather (
    id                  SERIAL PRIMARY KEY,
    airport             VARCHAR(10),
    timestamp           TIMESTAMP,
    temperature         FLOAT,
    wind_speed          FLOAT,
    visibility          INTEGER,
    rain                FLOAT,
    snow                FLOAT,
    weather_description VARCHAR(100),
    CONSTRAINT uq_weather_airport_ts UNIQUE (airport, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_weather_airport_ts   ON weather (airport, timestamp);
CREATE INDEX IF NOT EXISTS idx_flights_airport_date ON flights (departure_airport, date);
