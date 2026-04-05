from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
from fetch_flights import fetch_flights
from fetch_weather import fetch_weather
from load_postgres import insert_weather, insert_flight

AIRPORTS = ["KJFK", "KORD", "KDEN"]


@task(retries=3, retry_delay_seconds=60,
      cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def task_fetch_flights(airport):
    return fetch_flights(airport)


@task(retries=3, retry_delay_seconds=60)
def task_fetch_weather(airport):
    return fetch_weather(airport)


@task
def task_load_weather(weather):
    insert_weather(weather)


@task
def task_load_flights(flights):
    for f in flights:
        insert_flight(f)
    return len(flights)


@flow(name="flight-weather-pipeline", log_prints=True)
def pipeline():
    for airport in AIRPORTS:
        weather = task_fetch_weather(airport)
        task_load_weather(weather)
        flights = task_fetch_flights(airport)
        count = task_load_flights(flights)
        print(f"{airport}: {count} flights loaded")


if __name__ == "__main__":
    pipeline.serve(name="scheduled-run", interval=21600)  # every 6 hours
