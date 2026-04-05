import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry

BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "model" / "model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

AIRPORTS = {
    "JFK": {"name": "JOHN F. KENNEDY", "lat": 40.639, "lon": -73.779, "icao": "KJFK"},
    "ORD": {"name": "CHICAGO O'HARE",  "lat": 41.978, "lon": -87.904, "icao": "KORD"},
    "DEN": {"name": "DENVER INTL",     "lat": 39.856, "lon": -104.674, "icao": "KDEN"},
}

FORECAST_DAYS = 16


@st.cache_resource
def load_model():
    # pickle loads our own trained sklearn models — not untrusted external content
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def get_forecast_weather(lat: float, lon: float, target_dt: datetime) -> dict:
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["temperature_2m", "wind_speed_10m", "rain", "snowfall"],
        "wind_speed_unit": "kmh",
        "forecast_days": FORECAST_DAYS,
    }
    response = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
    hourly = response.Hourly()
    times = [
        datetime.utcfromtimestamp(hourly.Time() + i * hourly.Interval())
        for i in range(hourly.Variables(0).ValuesAsNumpy().shape[0])
    ]
    target_naive = target_dt.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    idx = min(range(len(times)), key=lambda i: abs((times[i] - target_naive).total_seconds()))
    return {
        "temperature": float(hourly.Variables(0).ValuesAsNumpy()[idx]),
        "wind_speed":  float(hourly.Variables(1).ValuesAsNumpy()[idx]),
        "rain":        float(hourly.Variables(2).ValuesAsNumpy()[idx]),
        "snow":        float(hourly.Variables(3).ValuesAsNumpy()[idx]),
    }


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FLIGHT DELAY PREDICTOR",
    page_icon="✈",
    layout="centered",
)

# ── Global CSS (injected into parent window via JS to avoid Streamlit text rendering bug) ──
import json as _json

_CSS = """
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.04) 2px,rgba(0,0,0,0.04) 4px);
    pointer-events: none;
    z-index: 9999;
}
.stApp { background: #0A0C10 !important; font-family: 'Space Mono', monospace !important; }
.main .block-container { max-width: 900px; padding: 0 1.5rem 5rem; }
#MainMenu, header[data-testid="stHeader"], footer, .stDeployButton { display: none !important; }
button[data-testid="baseButton-primary"] {
    background: #F59E0B !important; color: #0A0C10 !important;
    font-family: 'Space Mono', monospace !important; font-weight: 700 !important;
    letter-spacing: 0.1em !important; border: none !important; border-radius: 0 !important;
    text-transform: uppercase !important; transition: background .2s !important;
}
button[data-testid="baseButton-primary"]:hover { background: #FCD34D !important; }
button[data-testid="baseButton-secondary"] {
    background: #161920 !important; color: #7A7560 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important;
    letter-spacing: 0.06em !important; border: 1px solid rgba(245,158,11,0.2) !important;
    border-radius: 0 !important; text-transform: uppercase !important; transition: all .2s !important;
}
button[data-testid="baseButton-secondary"]:hover {
    border-color: rgba(245,158,11,0.5) !important; color: #F59E0B !important;
    background: rgba(245,158,11,0.06) !important;
}
input, select {
    background: #161920 !important; border: 1px solid rgba(245,158,11,0.2) !important;
    border-radius: 0 !important; color: #F3F0E8 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.85rem !important;
}
input:focus, select:focus { border-color: #F59E0B !important; box-shadow: 0 0 0 1px rgba(245,158,11,0.4) !important; }
div[data-testid="stDateInput"] label,
div[data-testid="stTimeInput"] label,
div[data-testid="stNumberInput"] label {
    font-family: 'Space Mono', monospace !important; font-size: 0.68rem !important;
    letter-spacing: 0.12em !important; color: #7A7560 !important; text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] button {
    border-radius: 0 !important; background: #1C2030 !important;
    color: #7A7560 !important; border: 1px solid rgba(245,158,11,0.2) !important;
}
hr { border-color: rgba(245,158,11,0.12) !important; }
@keyframes radarSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
.panel-label {
    font-family: 'Space Mono', monospace; font-size: 9px; letter-spacing: 0.16em;
    color: #7A7560; text-transform: uppercase; margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 10px;
}
.panel-label::before { content: ''; display: inline-block; width: 20px; height: 1px; background: #F59E0B; flex-shrink: 0; }
.cockpit-panel { background: #0F1117; border: 1px solid rgba(245,158,11,0.18); padding: 1.3rem 1.5rem; margin-bottom: 2px; }
.wx-strip { display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.18); margin-bottom: 2px; animation: fadeUp 0.35s ease; }
.wx-cell { background: #0F1117; padding: 1.3rem 0.5rem; text-align: center; }
.wx-ico  { font-size: 1rem; color: #7A7560; margin-bottom: 0.4rem; }
.wx-val  { font-family: 'Space Mono',monospace; font-size: 1.25rem; font-weight: 700; color: #F3F0E8; }
.wx-unit { font-family: 'Space Mono',monospace; font-size: 0.62rem; color: #7A7560; letter-spacing: .08em; margin-top: .2rem; }
.prob-panel { display: grid; grid-template-columns: 160px 1fr; gap: 1px; background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.18); margin-bottom: 2px; animation: fadeUp 0.5s ease; }
.prob-gauge-cell { background: #0F1117; padding: 1.6rem; display:flex; align-items:center; justify-content:center; }
.prob-info-cell  { background: #0F1117; padding: 1.6rem 2rem; display:flex; flex-direction:column; justify-content:center; }
.prob-status { font-family: 'Syne',sans-serif; font-size: 1.35rem; font-weight: 700; margin-bottom: .5rem; }
.prob-desc   { font-family: 'Space Mono',monospace; font-size: .7rem; color: #7A7560; line-height: 1.65; margin-bottom: .9rem; }
.prob-bar-wrap { height: 3px; background: rgba(245,158,11,0.1); overflow: hidden; }
.prob-bar-fill { height: 100%; transition: width 1s ease; }
.forecast-row-label { font-family: 'Space Mono',monospace; font-size: 9px; color: #7A7560; letter-spacing: .1em; text-transform: uppercase; background: #161920; border: 1px solid rgba(245,158,11,0.18); border-bottom: none; padding: 8px 14px; }
.factors-panel { border: 1px solid rgba(245,158,11,0.18); border-top: none; }
.factors-toggle { width: 100%; background: #161920; border: none; border-bottom: 1px solid rgba(245,158,11,0.18); color: #7A7560; font-family: 'Space Mono',monospace; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; padding: .9rem 1.4rem; cursor: pointer; display: flex; justify-content: space-between; align-items: center; transition: color .15s; }
.factors-toggle:hover { color: #F59E0B; }
.factors-toggle.open { color: #F59E0B; border-bottom-color: rgba(245,158,11,0.35); }
.factors-toggle .arr { transition: transform .35s ease; }
.factors-toggle.open .arr { transform: rotate(180deg); }
.factors-body { max-height: 0; overflow: hidden; transition: max-height .45s cubic-bezier(.4,0,.2,1); background: #0F1117; }
.factors-body.open { max-height: 600px; }
.factor-row { display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid rgba(245,158,11,0.07); font-family: 'Space Mono',monospace; font-size: .7rem; }
.factor-row:last-child { border-bottom: none; }
.f-name { color: #7A7560; padding: .55rem 1.4rem; }
.f-val  { color: #F3F0E8; font-weight: 700; padding: .55rem 1.4rem; letter-spacing: .04em; }
.model-note { font-family: 'Space Mono',monospace; font-size: .62rem; color: #3d3b2f; letter-spacing: .08em; text-transform: uppercase; padding: .7rem 1.4rem; border-top: 1px solid rgba(245,158,11,0.08); background: #161920; display: flex; align-items: center; gap: .5rem; }
.model-note::before { content: ''; display: inline-block; width: 4px; height: 4px; background: #F59E0B; border-radius: 50%; flex-shrink: 0; }
@media (max-width: 640px) { .prob-panel { grid-template-columns: 1fr; } .wx-strip { grid-template-columns: repeat(2,1fr); } }
"""

components.html(f"""<script>
(function() {{
  var p = window.parent.document;
  [{_json.dumps('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap')},
   {_json.dumps('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css')}
  ].forEach(function(href) {{
    if (!p.querySelector('link[href="'+href+'"]')) {{
      var l = p.createElement('link'); l.rel='stylesheet'; l.href=href;
      p.head.appendChild(l);
    }}
  }});
  if (!p.getElementById('cockpit-css')) {{
    var s = p.createElement('style'); s.id='cockpit-css';
    s.textContent = {_json.dumps(_CSS)};
    p.head.appendChild(s);
  }}
}})();
</script>""", height=0, scrolling=False)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:2.5rem 0 2rem;border-bottom:1px solid rgba(245,158,11,0.15);
            margin-bottom:1.5rem">
  <div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:.8rem">
      <span style="width:7px;height:7px;background:#10B981;border-radius:50%;
                   display:inline-block;animation:pulse 2s ease infinite"></span>
      <span style="font-family:'Space Mono',monospace;font-size:9px;
                   letter-spacing:.14em;color:#10B981;text-transform:uppercase">
        LIVE SYSTEM ONLINE
      </span>
    </div>
    <h1 style="font-family:'Syne',sans-serif;font-size:2.3rem;font-weight:800;
               letter-spacing:-.02em;margin:0 0 .5rem;line-height:1.1;color:#F3F0E8">
      FLIGHT <span style="color:#F59E0B">DELAY</span><br>PREDICTOR
    </h1>
    <p style="font-family:'Space Mono',monospace;font-size:10px;color:#7A7560;
              letter-spacing:.07em;text-transform:uppercase;margin:0">
      REAL BTS DATA &middot; OPEN-METEO FORECAST &middot; XGBOOST MODEL
    </p>
  </div>

  <!-- Radar SVG -->
  <div style="flex-shrink:0">
    <svg viewBox="0 0 80 80" width="90" height="90">
      <defs>
        <radialGradient id="rg" cx="40" cy="40" r="36" gradientUnits="userSpaceOnUse">
          <stop offset="0%"   stop-color="#F59E0B" stop-opacity=".8"/>
          <stop offset="100%" stop-color="#F59E0B" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <circle cx="40" cy="40" r="36" fill="none" stroke="#F59E0B" stroke-width=".7" opacity=".25"/>
      <circle cx="40" cy="40" r="24" fill="none" stroke="#F59E0B" stroke-width=".7" opacity=".2"/>
      <circle cx="40" cy="40" r="12" fill="none" stroke="#F59E0B" stroke-width=".7" opacity=".15"/>
      <line x1="4" y1="40" x2="76" y2="40" stroke="#F59E0B" stroke-width=".5" opacity=".12"/>
      <line x1="40" y1="4" x2="40" y2="76" stroke="#F59E0B" stroke-width=".5" opacity=".12"/>
      <path d="M40,40 L40,4 A36,36 0 0,1 65.5,22 Z" fill="url(#rg)"
            style="transform-origin:40px 40px;animation:radarSpin 4s linear infinite"/>
      <circle cx="40" cy="40" r="2.5" fill="#F59E0B"/>
      <circle cx="18" cy="24" r="1.8" fill="#10B981" opacity=".7"/>
      <circle cx="58" cy="16" r="1.2" fill="#F59E0B" opacity=".5"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "selected_airport" not in st.session_state:
    st.session_state["selected_airport"] = "JFK"
if "dep_hour" not in st.session_state:
    st.session_state["dep_hour"] = datetime.utcnow().hour

# ── Airport + Datetime panels (side by side) ──────────────────────────────────
left_col, right_col = st.columns(2, gap="small")

with left_col:
    st.markdown('<div class="cockpit-panel" style="height:100%">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">ORIGIN AIRPORT</div>', unsafe_allow_html=True)
    ap_cols = st.columns(3, gap="small")
    for col, (code, info) in zip(ap_cols, AIRPORTS.items()):
        is_active = st.session_state["selected_airport"] == code
        label = f"**{code}**  \n{info['name']}"
        if col.button(label, key=f"ap_{code}",
                      type="primary" if is_active else "secondary",
                      use_container_width=True):
            st.session_state["selected_airport"] = code
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="cockpit-panel" style="height:100%">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">DEPARTURE DATE</div>', unsafe_allow_html=True)
    today    = date.today()
    max_date = today + timedelta(days=FORECAST_DAYS - 1)
    dep_date = st.date_input(
        "Date",
        value=today + timedelta(days=1),
        min_value=today,
        max_value=max_date,
        help=f"Open-Meteo provides hourly forecasts up to {FORECAST_DAYS} days ahead.",
        label_visibility="collapsed",
    )
    st.markdown('<div class="panel-label" style="margin-top:1rem">DEPARTURE TIME (UTC)</div>', unsafe_allow_html=True)
    preset_cols = st.columns(4, gap="small")
    for col, (label, h) in zip(preset_cols, [("09:00",9),("12:00",12),("16:00",16),("18:00",18)]):
        is_active = st.session_state["dep_hour"] == h
        if col.button(label, key=f"t_{h}",
                      type="primary" if is_active else "secondary",
                      use_container_width=True):
            st.session_state["dep_hour"] = h
            st.rerun()
    st.selectbox(
        "Hour (UTC)",
        options=list(range(24)),
        format_func=lambda h: f"{h:02d}:00 UTC",
        label_visibility="collapsed",
        key="dep_hour",
    )
    st.markdown('</div>', unsafe_allow_html=True)

dep_hour = st.session_state["dep_hour"]
dep_dt   = datetime(dep_date.year, dep_date.month, dep_date.day, dep_hour)

# ── Known delay context ───────────────────────────────────────────────────────
st.markdown('<div class="cockpit-panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-label">KNOWN DELAY CONTEXT <span style="margin-left:.5rem;font-size:8px;color:#4a4637">(OPTIONAL — ENTER IF VISIBLE IN AIRLINE APP)</span></div>', unsafe_allow_html=True)

dc1, dc2 = st.columns(2, gap="medium")
weather_delay = dc1.number_input("Weather delay (min)",         0, 500, 0)
carrier_delay = dc2.number_input("Carrier / airline delay (min)", 0, 500, 0)
nas_delay     = dc1.number_input("Air system / NAS delay (min)", 0, 500, 0)
late_ac_delay = dc2.number_input("Late incoming aircraft (min)", 0, 500, 0)

st.markdown('</div>', unsafe_allow_html=True)

# ── CTA ───────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
predict = st.button("⟶  RUN DELAY ANALYSIS", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if predict:
    selected_code = st.session_state["selected_airport"]
    airport_info  = AIRPORTS[selected_code]

    with st.spinner("FETCHING FORECAST…"):
        try:
            w = get_forecast_weather(airport_info["lat"], airport_info["lon"], dep_dt)
        except Exception as e:
            st.error(f"Weather fetch failed: {e}")
            st.stop()

    # Weather strip
    st.markdown(f"""
    <div class="forecast-row-label">
      &#x25BA; FORECAST &middot; {dep_date.strftime('%B %d, %Y').upper()} AT {dep_hour:02d}:00 UTC &middot; {airport_info['icao']}
    </div>
    <div class="wx-strip">
      <div class="wx-cell">
        <div class="wx-ico"><i class="fa-solid fa-temperature-half"></i></div>
        <div class="wx-val">{w['temperature']:.1f}</div>
        <div class="wx-unit">TEMP &deg;C</div>
      </div>
      <div class="wx-cell">
        <div class="wx-ico"><i class="fa-solid fa-wind"></i></div>
        <div class="wx-val">{w['wind_speed']:.1f}</div>
        <div class="wx-unit">WIND KM/H</div>
      </div>
      <div class="wx-cell">
        <div class="wx-ico"><i class="fa-solid fa-cloud-rain"></i></div>
        <div class="wx-val">{w['rain']:.1f}</div>
        <div class="wx-unit">RAIN MM/H</div>
      </div>
      <div class="wx-cell">
        <div class="wx-ico"><i class="fa-solid fa-snowflake"></i></div>
        <div class="wx-val">{w['snow']:.1f}</div>
        <div class="wx-unit">SNOW MM/H</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Prediction
    model, scaler = load_model()
    features = np.array([[
        w["temperature"], w["wind_speed"], w["rain"], w["snow"],
        dep_hour, dep_date.month,
        weather_delay, carrier_delay, nas_delay, late_ac_delay,
    ]])
    prob     = model.predict_proba(scaler.transform(features))[0][1]
    prob_pct = prob * 100

    if prob > 0.55:
        g_color  = "#EF4444"
        status   = "HIGH DELAY RISK"
        desc     = "Significant delay likely. Check alternatives or plan extra buffer time."
    elif prob > 0.25:
        g_color  = "#F59E0B"
        status   = "MODERATE DELAY RISK"
        desc     = "Delay possible. Monitor your flight status closer to departure."
    else:
        g_color  = "#10B981"
        status   = "LOW DELAY RISK"
        desc     = "Conditions look favorable. On-time departure is likely."

    # Gauge
    circumference = 283  # 2 * pi * 45
    offset = circumference - (prob_pct / 100) * circumference

    st.markdown(f"""
    <style>
      @keyframes gaugeIn {{
        from {{ stroke-dashoffset: {circumference}; }}
        to   {{ stroke-dashoffset: {offset:.2f}; }}
      }}
      #g-arc {{ animation: gaugeIn 1.3s cubic-bezier(.4,0,.2,1) forwards; }}
    </style>
    <div class="prob-panel">
      <div class="prob-gauge-cell">
        <svg viewBox="0 0 120 120" width="130" height="130">
          <circle cx="60" cy="60" r="45" fill="none"
                  stroke="rgba(245,158,11,0.08)" stroke-width="9"/>
          <circle id="g-arc" cx="60" cy="60" r="45" fill="none"
                  stroke="{g_color}" stroke-width="9"
                  stroke-dasharray="{circumference}"
                  stroke-dashoffset="{circumference}"
                  stroke-linecap="round"
                  transform="rotate(-90 60 60)"/>
          <text x="60" y="56" text-anchor="middle"
                fill="{g_color}" font-family="Space Mono,monospace"
                font-size="20" font-weight="700">{prob_pct:.0f}%</text>
          <text x="60" y="72" text-anchor="middle"
                fill="#7A7560" font-family="Space Mono,monospace"
                font-size="7" letter-spacing="2">DELAY PROB</text>
        </svg>
      </div>
      <div class="prob-info-cell">
        <div class="prob-status" style="color:{g_color}">{status}</div>
        <div class="prob-desc">{desc}</div>
        <div class="prob-bar-wrap">
          <div class="prob-bar-fill"
               style="width:{prob_pct:.1f}%;background:{g_color}"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Factor rows
    def frow(label, val):
        return f'<div class="factor-row"><div class="f-name">{label}</div><div class="f-val">{val}</div></div>'

    rows = "".join([
        frow("TEMPERATURE",         f"{w['temperature']:.1f} &deg;C"),
        frow("WIND SPEED",          f"{w['wind_speed']:.1f} KM/H"),
        frow("RAINFALL",            f"{w['rain']:.1f} MM/H"),
        frow("SNOWFALL",            f"{w['snow']:.1f} MM/H"),
        frow("DEPARTURE HOUR",      f"{dep_hour:02d}:00 UTC"),
        frow("MONTH",               dep_date.strftime("%B").upper()),
        frow("WEATHER DELAY",       f"{weather_delay} MIN"),
        frow("CARRIER DELAY",       f"{carrier_delay} MIN"),
        frow("AIR SYSTEM DELAY",    f"{nas_delay} MIN"),
        frow("LATE AIRCRAFT DELAY", f"{late_ac_delay} MIN"),
    ])

    st.markdown(f"""
    <div class="factors-panel">
      <button class="factors-toggle" onclick="
        this.classList.toggle('open');
        this.nextElementSibling.classList.toggle('open');
      ">
        <span>WHAT DROVE THIS PREDICTION</span>
        <span class="arr">&#x25BC;</span>
      </button>
      <div class="factors-body">
        {rows}
        <div class="model-note">
          XGBOOST &middot; TRAINED ON 2015 BTS FLIGHT DATA (JFK, ORD, DEN) &middot; SELECTED BY HIGHEST ROC-AUC
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
