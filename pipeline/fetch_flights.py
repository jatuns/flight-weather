import requests
from datetime import datetime

OPENSKY_URL = "https://opensky-network.org/api/flights/departure"

# BTS IATA → ICAO mapping for the airports in scope
IATA_TO_ICAO = {
    "JFK": "KJFK",
    "ORD": "KORD",
    "DEN": "KDEN",
}


def fetch_flights(airport: str, hours_back: int = 6) -> list:
    """
    Fetch recent departures from OpenSky for live pipeline.
    airport: ICAO code (e.g. KJFK)
    delay_minutes is always NULL here — BTS seed data provides historical delays.
    """
    now = int(datetime.utcnow().timestamp())
    begin = now - (hours_back * 3600)

    resp = requests.get(OPENSKY_URL, params={
        "airport": airport, "begin": begin, "end": now
    }, timeout=30)
    resp.raise_for_status()

    result = []
    for f in resp.json():
        actual_dep = datetime.utcfromtimestamp(f["firstSeen"]) if f.get("firstSeen") else None
        result.append({
            "icao24": f.get("icao24"),
            "date": actual_dep,
            "actual_departure": actual_dep,
            "scheduled_departure": None,
            "departure_airport": airport,
            "arrival_airport": f.get("estArrivalAirport"),
            "delay_minutes": None,
            "weather_delay_min": None,
            "carrier_delay_min": None,
            "nas_delay_min": None,
            "security_delay_min": None,
            "late_aircraft_delay_min": None,
            "flight_status": "departed",
            "airline": (f.get("callsign") or "")[:100]
        })
    return result
