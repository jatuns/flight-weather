-- 1. Rain vs no-rain average delay
SELECT
    CASE WHEN w.rain > 0 THEN 'Rainy' ELSE 'Clear' END AS condition,
    COUNT(*) AS flights,
    ROUND(AVG(f.delay_minutes), 2) AS avg_delay_minutes
FROM flights f
LEFT JOIN weather w
    ON f.departure_airport = w.airport
    AND DATE_TRUNC('hour', f.date) = DATE_TRUNC('hour', w.timestamp)
WHERE f.delay_minutes IS NOT NULL
GROUP BY 1;

-- 2. Top airports by delay rate (min 50 flights)
SELECT
    departure_airport,
    COUNT(*) AS total_flights,
    ROUND(AVG(delay_minutes), 2) AS avg_delay,
    ROUND(100.0 * SUM(CASE WHEN delay_minutes > 15 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_delayed
FROM flights
WHERE delay_minutes IS NOT NULL
GROUP BY departure_airport
HAVING COUNT(*) >= 50
ORDER BY avg_delay DESC;

-- 3. Delay by hour of day
SELECT
    EXTRACT(HOUR FROM date) AS hour,
    ROUND(AVG(delay_minutes), 2) AS avg_delay
FROM flights
WHERE delay_minutes IS NOT NULL
GROUP BY 1 ORDER BY 1;

-- 4. Delay by month (seasonality)
SELECT
    EXTRACT(MONTH FROM date) AS month,
    ROUND(AVG(delay_minutes), 2) AS avg_delay
FROM flights
WHERE delay_minutes IS NOT NULL
GROUP BY 1 ORDER BY 1;

-- 5. Wind speed buckets vs delay
SELECT
    CASE
        WHEN w.wind_speed < 20 THEN '0-20 km/h'
        WHEN w.wind_speed < 40 THEN '20-40 km/h'
        WHEN w.wind_speed < 60 THEN '40-60 km/h'
        ELSE '60+ km/h'
    END AS wind_bucket,
    COUNT(*) AS flights,
    ROUND(AVG(f.delay_minutes), 2) AS avg_delay
FROM flights f
LEFT JOIN weather w
    ON f.departure_airport = w.airport
    AND DATE_TRUNC('hour', f.date) = DATE_TRUNC('hour', w.timestamp)
WHERE f.delay_minutes IS NOT NULL AND w.wind_speed IS NOT NULL
GROUP BY 1
ORDER BY MIN(w.wind_speed);

-- 6. Delay cause breakdown
SELECT
    ROUND(AVG(weather_delay_min), 2)       AS avg_weather_delay,
    ROUND(AVG(carrier_delay_min), 2)       AS avg_carrier_delay,
    ROUND(AVG(nas_delay_min), 2)           AS avg_nas_delay,
    ROUND(AVG(late_aircraft_delay_min), 2) AS avg_late_aircraft_delay
FROM flights
WHERE delay_minutes > 15;

-- 7. Weather delay vs actual weather conditions
SELECT
    CASE WHEN w.rain > 2 OR w.snow > 0.5 THEN 'Bad weather' ELSE 'Normal' END AS weather_condition,
    COUNT(*) AS flights,
    ROUND(AVG(f.weather_delay_min), 2) AS avg_weather_delay_min,
    ROUND(AVG(f.delay_minutes), 2)     AS avg_total_delay
FROM flights f
LEFT JOIN weather w
    ON f.departure_airport = w.airport
    AND DATE_TRUNC('hour', f.date) = DATE_TRUNC('hour', w.timestamp)
WHERE f.delay_minutes IS NOT NULL
GROUP BY 1;
