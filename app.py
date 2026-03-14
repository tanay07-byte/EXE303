"""
PyClimaExplorer — TECHNEX'26 Hackathon
Interactive Climate Data Visualizer

ONE location selector in the sidebar controls ALL tabs simultaneously:
  ✅ Spatial Heatmap   — pin on map + surrounding heatmap context
  ✅ Time-Series       — monthly/annual for the selected location
  ✅ 3D Globe          — selected location highlighted on globe
  ✅ Compare Mode      — selected location vs a second location
  ✅ Guided Story      — anomaly charts centred on selected location

Stack: Python · Streamlit · Plotly · NumPy · Pandas · Xarray
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr
from sklearn.linear_model import LinearRegression
import warnings
import time
warnings.filterwarnings("ignore")

# ── Safe line helpers (avoid Plotly add_vline/add_hline Timestamp/string bug) ──
def vline(fig, x, color="rgba(255,255,255,0.4)", dash="dash", width=1.5,
          label="", label_color="white", font_size=10):
    """Add a vertical reference line using shapes+annotations (no arithmetic on x)."""
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=x, x1=x, y0=0, y1=1,
                  line=dict(color=color, width=width, dash=dash))
    if label:
        fig.add_annotation(x=x, yref="paper", y=1.02, text=label,
                           showarrow=False, xanchor="left",
                           font=dict(color=label_color, size=font_size))

def hline(fig, y, color="rgba(255,255,255,0.4)", dash="dash", width=1.5,
          label="", label_color="white", font_size=10):
    """Add a horizontal reference line using shapes+annotations."""
    fig.add_shape(type="line", xref="paper", yref="y",
                  x0=0, x1=1, y0=y, y1=y,
                  line=dict(color=color, width=width, dash=dash))
    if label:
        fig.add_annotation(xref="paper", x=1.01, y=y, text=label,
                           showarrow=False, xanchor="left",
                           font=dict(color=label_color, size=font_size))

def generate_response(prompt):
    """Enhanced rule-based chatbot for climate app with broader understanding."""
    prompt = prompt.lower().strip()
    
    # Keywords and responses
    responses = {
        # Temperature related
        ("temperature", "temp", "hot", "cold", "warm", "heat"): 
            "Temperature shows surface air temperature in °C. Check the Spatial Heatmap or Time-Series tabs to see values for your selected location. Anomalies show deviation from the 1950-1980 baseline.",
        
        # Anomaly related
        ("anomaly", "deviation", "normal", "baseline", "average"): 
            "Anomalies measure how much climate variables differ from the long-term average (1950-1980 baseline). Positive anomalies mean warmer/wetter than normal, negative means cooler/drier. Look at the anomaly charts in various tabs.",
        
        # Precipitation related
        ("precipitation", "rain", "snow", "drought", "flood", "water"): 
            "Precipitation is measured in mm/day and includes rain, snow, etc. The app shows how much water falls at different locations. Anomalies can indicate droughts or floods.",
        
        # Wind related
        ("wind", "speed", "blow", "storm"): 
            "Wind speed is measured in m/s. It shows how fast air moves at the surface. Higher speeds often occur in open areas or during storms.",
        
        # App usage
        ("how", "use", "navigate", "tab", "work", "start", "begin"): 
            "Select a location in the sidebar to update all tabs. Each tab shows different views: Spatial Heatmap (global map), Time-Series (trends over time), 3D Globe (interactive globe), Compare Mode (compare two periods), Guided Story (climate narrative), and this AI Assistant.",
        
        # Location related
        ("location", "place", "city", "country", "select", "choose"): 
            "Use the search box in the sidebar to find cities or regions. You can also adjust latitude/longitude sliders for precise locations. One location controls all tabs simultaneously.",
        
        # Data/time related
        ("year", "month", "time", "date", "period", "change"): 
            "Use the year and month sliders in the sidebar to explore different time periods. The data spans 1950-2024, showing climate trends and anomalies.",
        
        # Help/general
        ("help", "what", "explain", "tell", "show"): 
            "I'm here to help with climate data! Ask about temperature, precipitation, wind, anomalies, or how to use the app. You can also ask about specific features or data interpretation.",
        
        # Climate change
        ("climate", "change", "global", "warming", "trend"): 
            "The app shows climate trends from 1950-2024. Global temperatures have risen about 0.02°C per year. Check the Guided Story tab for a narrative of key climate events.",
    }
    
    # Check for matches
    for keywords, response in responses.items():
        if any(keyword in prompt for keyword in keywords):
            return response
    
    # Fallback for unrecognized questions
    return "I can help with questions about climate data, anomalies, temperature, precipitation, wind, and how to use this app. Try asking something like 'What is an anomaly?' or 'How do I change the location?'"



#  PAGE CONFIG

st.set_page_config(
    page_title="PyClimaExplorer — TECHNEX'26",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NASA_IMAGE_URL = "https://images.unsplash.com/photo-1634176866089-b633f4aec882?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8ZWFydGh8ZW58MHx8MHx8fDA=3&s=8c9b1c8e7a0c9b1e5f8d9c9a1b2c3d4"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif !important; }}

.stApp {{
    background-image:
        linear-gradient(135deg,rgba(5,10,20,0.65) 0%,rgba(8,16,32,0.58) 50%,rgba(5,10,20,0.70) 100%),
        url("{NASA_IMAGE_URL}");
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}}
body::after {{
    content:""; position:fixed; top:0; left:0; width:100%; height:100%;
    background-image:
        radial-gradient(1.5px 1.5px at 20px 30px,rgba(255,255,255,0.9),transparent),
        radial-gradient(1px 1px at 40px 70px,rgba(255,255,255,0.7),transparent),
        radial-gradient(1.5px 1.5px at 90px 40px,rgba(255,255,255,0.8),transparent),
        radial-gradient(1px 1px at 160px 25px,rgba(255,255,255,0.5),transparent),
        radial-gradient(1.5px 1.5px at 200px 60px,rgba(255,255,255,0.9),transparent);
    background-repeat:repeat; background-size:250px 200px;
    animation:starsMove 180s linear infinite; opacity:0.3; z-index:-1; pointer-events:none;
}}
@keyframes starsMove {{ from{{transform:translateY(0)}} to{{transform:translateY(-2000px)}} }}

.main-header,.metric-card,.story-card,.diff-banner,.anomaly-card {{
    background:rgba(10,18,35,0.55) !important;
    backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
    border:1px solid rgba(255,255,255,0.12) !important;
    box-shadow:0 8px 32px rgba(0,0,0,0.55);
}}
section[data-testid="stSidebar"] {{
    background:rgba(6,12,26,0.80) !important;
    backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
    border-right:1px solid rgba(0,212,170,0.18) !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background:rgba(10,18,35,0.65) !important; backdrop-filter:blur(12px);
    border-radius:10px; padding:4px; gap:4px;
    border:1px solid rgba(255,255,255,0.06);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:8px; color:#64748b; font-weight:500; font-size:13px; padding:8px 18px;
}}
.stTabs [aria-selected="true"] {{
    background:rgba(0,212,170,0.15) !important;
    color:#00d4aa !important;
    border:1px solid rgba(0,212,170,0.3) !important;
}}
.main-header {{
    background:linear-gradient(90deg,rgba(0,212,170,0.15) 0%,rgba(59,130,246,0.08) 100%) !important;
    border:1px solid rgba(0,212,170,0.25) !important;
    border-radius:14px; padding:18px 28px; margin-bottom:24px;
    display:flex; align-items:center; justify-content:space-between;
}}
.header-logo {{ font-size:26px; font-weight:700; color:#00d4aa; text-shadow:0 0 20px rgba(0,212,170,0.6); }}
.header-badge {{
    background:rgba(0,212,170,0.15); border:1px solid rgba(0,212,170,0.3);
    color:#00d4aa; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:600; letter-spacing:1px;
}}
.loc-badge {{
    background:rgba(0,212,170,0.12); border:1px solid rgba(0,212,170,0.3);
    border-radius:10px; padding:10px 16px; margin-bottom:14px; text-align:center;
}}
.loc-name  {{ color:#00d4aa; font-size:15px; font-weight:700; }}
.loc-coords{{ color:#64748b; font-size:11px; margin-top:3px; }}
.metric-card {{ border-radius:12px; padding:18px 20px; position:relative; overflow:hidden; }}
.metric-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:12px 12px 0 0; }}
.metric-card.accent-teal::before   {{ background:#00d4aa; }}
.metric-card.accent-blue::before   {{ background:#3b82f6; }}
.metric-card.accent-orange::before {{ background:#f59e0b; }}
.metric-card.accent-red::before    {{ background:#ef4444; }}
.metric-label {{ font-size:11px; color:#94a3b8; font-weight:600; letter-spacing:1px; text-transform:uppercase; }}
.metric-value {{ font-size:26px; font-weight:700; color:#e2e8f0; margin:4px 0 2px; }}
.metric-sub   {{ font-size:11px; color:#64748b; }}
.sidebar-section-title {{
    font-size:11px; font-weight:600; letter-spacing:1.5px; color:#64748b;
    text-transform:uppercase; margin-bottom:12px; padding-bottom:8px;
    border-bottom:1px solid rgba(255,255,255,0.06);
}}
.story-card {{
    border-left:3px solid #00d4aa !important; border-radius:0 12px 12px 0;
    padding:24px; margin-bottom:16px; animation:fadeIn 0.4s ease;
}}
.story-year    {{ color:#00d4aa; font-size:12px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px; }}
.story-heading {{ font-size:22px; font-weight:700; color:#e2e8f0; margin-bottom:10px; }}
.story-body    {{ font-size:14px; color:#cbd5e1; line-height:1.75; }}
.anomaly-card  {{
    background:linear-gradient(135deg,rgba(239,68,68,0.12) 0%,rgba(245,158,11,0.06) 100%) !important;
    border:1px solid rgba(239,68,68,0.3) !important;
    border-radius:10px; padding:14px 18px; margin-top:14px;
}}
.anomaly-title {{ font-weight:700; font-size:13px; color:#fca5a5; margin-bottom:6px; }}
.anomaly-desc  {{ font-size:12px; color:#94a3b8; line-height:1.6; }}
.chip {{ display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; margin-right:6px; }}
.chip-hot    {{ background:rgba(239,68,68,0.2);  color:#fca5a5; }}
.chip-record {{ background:rgba(245,158,11,0.2); color:#fcd34d; }}
.chip-cold   {{ background:rgba(34,211,238,0.15); color:#67e8f9; }}
.diff-banner {{ border-radius:10px; padding:20px 28px; text-align:center; margin-top:16px; }}
.diff-num    {{ font-size:36px; font-weight:700; color:#f59e0b; }}
.diff-label  {{ font-size:12px; color:#94a3b8; margin-top:4px; }}
@keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}
hr {{ border-color:rgba(255,255,255,0.06) !important; }}
#MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)

#  WORLD CITY DATABASE

WORLD_CITIES = {
    "📍 Varanasi, India":            (25.3,   83.0),
    "📍 Mumbai, India":              (19.1,   72.9),
    "📍 Delhi, India":               (28.6,   77.2),
    "📍 Chennai, India":             (13.1,   80.3),
    "📍 Kolkata, India":             (22.6,   88.4),
    "📍 Bengaluru, India":           (12.9,   77.6),
    "📍 Hyderabad, India":           (17.4,   78.5),
    "📍 Jaipur, India":              (26.9,   75.8),
    "📍 Lucknow, India":             (26.8,   80.9),
    "📍 Kochi, India":               ( 9.9,   76.3),
    "📍 Guwahati, India":            (26.2,   91.7),
    "📍 Shimla, India":              (31.1,   77.2),
    "📍 Leh (Himalayas), India":     (34.2,   77.6),
    "📍 Dhaka, Bangladesh":          (23.7,   90.4),
    "📍 Karachi, Pakistan":          (24.9,   67.0),
    "📍 Islamabad, Pakistan":        (33.7,   73.1),
    "📍 Kathmandu, Nepal":           (27.7,   85.3),
    "📍 Colombo, Sri Lanka":         ( 6.9,   79.9),
    "📍 Kabul, Afghanistan":         (34.5,   69.2),
    "📍 Beijing, China":             (39.9,  116.4),
    "📍 Shanghai, China":            (31.2,  121.5),
    "📍 Hong Kong, China":           (22.3,  114.2),
    "📍 Guangzhou, China":           (23.1,  113.3),
    "📍 Chengdu, China":             (30.7,  104.1),
    "📍 Lhasa, Tibet":               (29.7,   91.1),
    "📍 Harbin, China":              (45.8,  126.6),
    "📍 Tokyo, Japan":               (35.7,  139.7),
    "📍 Osaka, Japan":               (34.7,  135.5),
    "📍 Sapporo, Japan":             (43.1,  141.4),
    "📍 Seoul, South Korea":         (37.6,  127.0),
    "📍 Ulaanbaatar, Mongolia":      (47.9,  106.9),
    "📍 Bangkok, Thailand":          (13.8,  100.5),
    "📍 Singapore":                  ( 1.3,  103.8),
    "📍 Kuala Lumpur, Malaysia":     ( 3.1,  101.7),
    "📍 Jakarta, Indonesia":         (-6.2,  106.8),
    "📍 Bali, Indonesia":            (-8.7,  115.2),
    "📍 Manila, Philippines":        (14.6,  121.0),
    "📍 Hanoi, Vietnam":             (21.0,  105.8),
    "📍 Ho Chi Minh City, Vietnam":  (10.8,  106.7),
    "📍 Yangon, Myanmar":            (16.9,   96.2),
    "📍 Dubai, UAE":                 (25.2,   55.3),
    "📍 Riyadh, Saudi Arabia":       (24.7,   46.7),
    "📍 Tehran, Iran":               (35.7,   51.4),
    "📍 Baghdad, Iraq":              (33.3,   44.4),
    "📍 Istanbul, Turkey":           (41.0,   29.0),
    "📍 Ankara, Turkey":             (40.0,   32.9),
    "📍 Jerusalem, Israel":          (31.8,   35.2),
    "📍 Amman, Jordan":              (32.0,   36.0),
    "📍 Doha, Qatar":                (25.3,   51.5),
    "📍 Muscat, Oman":               (23.6,   58.6),
    "📍 Tashkent, Uzbekistan":       (41.3,   69.3),
    "📍 Almaty, Kazakhstan":         (43.3,   76.9),
    "📍 Baku, Azerbaijan":           (40.4,   49.9),
    "📍 Tbilisi, Georgia":           (41.7,   44.8),
    "📍 Moscow, Russia":             (55.8,   37.6),
    "📍 St Petersburg, Russia":      (59.9,   30.3),
    "📍 Novosibirsk, Russia":        (55.0,   82.9),
    "📍 Yakutsk, Russia":            (62.0,  129.7),
    "📍 Murmansk, Russia":           (68.9,   33.1),
    "📍 London, UK":                 (51.5,   -0.1),
    "📍 Edinburgh, Scotland":        (56.0,   -3.2),
    "📍 Dublin, Ireland":            (53.3,   -6.3),
    "📍 Paris, France":              (48.9,    2.4),
    "📍 Berlin, Germany":            (52.5,   13.4),
    "📍 Munich, Germany":            (48.1,   11.6),
    "📍 Rome, Italy":                (41.9,   12.5),
    "📍 Madrid, Spain":              (40.4,   -3.7),
    "📍 Barcelona, Spain":           (41.4,    2.2),
    "📍 Amsterdam, Netherlands":     (52.4,    4.9),
    "📍 Vienna, Austria":            (48.2,   16.4),
    "📍 Warsaw, Poland":             (52.2,   21.0),
    "📍 Stockholm, Sweden":          (59.3,   18.1),
    "📍 Oslo, Norway":               (59.9,   10.8),
    "📍 Helsinki, Finland":          (60.2,   24.9),
    "📍 Copenhagen, Denmark":        (55.7,   12.6),
    "📍 Reykjavik, Iceland":         (64.1,  -21.9),
    "📍 Athens, Greece":             (38.0,   23.7),
    "📍 Kyiv, Ukraine":              (50.5,   30.5),
    "📍 Lisbon, Portugal":           (38.7,   -9.1),
    "📍 Zurich, Switzerland":        (47.4,    8.5),
    "📍 Cairo, Egypt":               (30.1,   31.2),
    "📍 Lagos, Nigeria":             ( 6.5,    3.4),
    "📍 Nairobi, Kenya":             (-1.3,   36.8),
    "📍 Johannesburg, S. Africa":    (-26.2,  28.0),
    "📍 Cape Town, S. Africa":       (-33.9,  18.4),
    "📍 Casablanca, Morocco":        (33.6,   -7.6),
    "📍 Addis Ababa, Ethiopia":      ( 9.0,   38.7),
    "📍 Dar es Salaam, Tanzania":    (-6.8,   39.3),
    "📍 Kinshasa, DR Congo":         (-4.3,   15.3),
    "📍 Accra, Ghana":               ( 5.6,   -0.2),
    "📍 Dakar, Senegal":             (14.7,  -17.4),
    "📍 Algiers, Algeria":           (36.7,    3.1),
    "📍 Lusaka, Zambia":             (-15.4,  28.3),
    "📍 Kampala, Uganda":            ( 0.3,   32.6),
    "📍 New York City, USA":         (40.7,  -74.0),
    "📍 Los Angeles, USA":           (34.1, -118.2),
    "📍 Chicago, USA":               (41.9,  -87.6),
    "📍 Houston, USA":               (29.8,  -95.4),
    "📍 Miami, USA":                 (25.8,  -80.2),
    "📍 Seattle, USA":               (47.6, -122.3),
    "📍 Denver, USA":                (39.7, -104.9),
    "📍 Phoenix, USA":               (33.4, -112.1),
    "📍 Anchorage, Alaska, USA":     (61.2, -149.9),
    "📍 Honolulu, Hawaii, USA":      (21.3, -157.8),
    "📍 Toronto, Canada":            (43.7,  -79.4),
    "📍 Vancouver, Canada":          (49.3, -123.1),
    "📍 Montreal, Canada":           (45.5,  -73.6),
    "📍 Yellowknife, Canada":        (62.5, -114.4),
    "📍 Mexico City, Mexico":        (19.4,  -99.1),
    "📍 Havana, Cuba":               (23.1,  -82.4),
    "📍 Panama City, Panama":        ( 9.0,  -79.5),
    "📍 Sao Paulo, Brazil":          (-23.5, -46.6),
    "📍 Rio de Janeiro, Brazil":     (-22.9, -43.2),
    "📍 Manaus (Amazon), Brazil":    (-3.1,  -60.0),
    "📍 Buenos Aires, Argentina":    (-34.6, -58.4),
    "📍 Lima, Peru":                 (-12.0, -77.0),
    "📍 Bogota, Colombia":           ( 4.7,  -74.1),
    "📍 Santiago, Chile":            (-33.5, -70.6),
    "📍 Caracas, Venezuela":         (10.5,  -66.9),
    "📍 Quito, Ecuador":             (-0.2,  -78.5),
    "📍 La Paz, Bolivia":            (-16.5, -68.1),
    "📍 Patagonia (south tip)":      (-51.7, -72.5),
    "📍 Sydney, Australia":          (-33.9, 151.2),
    "📍 Melbourne, Australia":       (-37.8, 145.0),
    "📍 Perth, Australia":           (-31.9, 115.9),
    "📍 Darwin, Australia":          (-12.5, 130.8),
    "📍 Alice Springs, Australia":   (-23.7, 133.9),
    "📍 Auckland, New Zealand":      (-36.9, 174.8),
    "📍 Wellington, New Zealand":    (-41.3, 174.8),
    "📍 Suva, Fiji":                 (-18.1, 178.4),
    "🧊 Arctic (North Pole)":        (88.0,   0.0),
    "❄️ Antarctica (South Pole)":    (-88.0,  0.0),
    "🧊 Greenland (Nuuk)":           (64.2,  -51.7),
    "🧊 Siberia (coldest region)":   (63.0,  127.0),
    "🏔️ Himalayas (Mt. Everest)":   (28.0,   86.9),
    "🌊 North Atlantic Ocean":       (40.0,  -40.0),
    "🌊 Pacific Ocean (Central)":    ( 0.0, -160.0),
    "🌊 Indian Ocean (Central)":     (-10.0,  70.0),
    "🌵 Atacama Desert, Chile":      (-24.5,  -69.2),
    "🏜️ Sahara Desert, Algeria":    (23.0,   13.0),
    "🌿 Amazon Rainforest, Brazil":  (-3.5,  -60.0),
    "🔥 Death Valley, USA":          (36.5, -116.9),
}

EMOJI_STRIP = str.maketrans("", "", "📍🧊❄️🏔️🌊🌵🏜️🌿🔥✏️")
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


#  DATA GENERATION

@st.cache_data
def generate_synthetic_dataset():
    rng   = np.random.default_rng(42)
    lats  = np.arange(-87.5, 90, 5.0)
    lons  = np.arange(-177.5, 180, 5.0)
    years = np.arange(1950, 2025)
    months= np.arange(1, 13)
    lat_grid, _ = np.meshgrid(lats, lons, indexing='ij')

    def make_temp(lat, lon, year, month):
        lf     = np.cos(np.radians(lat))
        season = np.sin((month - 3) * np.pi / 6) * 15 * lf
        trend  = (year - 1950) * 0.02
        noise  = np.random.default_rng(int(abs(lat*100 + lon*10 + year + month))).normal(0, 3)
        return 30*lf - 15 + season + trend + noise

    T = np.zeros((len(lats), len(lons), len(years), len(months)))
    for yi, y in enumerate(years):
        for mi, m in enumerate(months):
            for li, la in enumerate(lats):
                for ci, lo in enumerate(lons):
                    T[li, ci, yi, mi] = make_temp(la, lo, y, m)

    P = np.abs(T*0.08 + rng.normal(0,1,T.shape)) * np.cos(np.radians(lat_grid))[:,:,np.newaxis,np.newaxis]
    P = np.clip(P, 0, None)
    W = 5 + np.abs(np.radians(lat_grid[:,:,np.newaxis,np.newaxis]))*5 + rng.normal(0,2,T.shape)
    W = np.clip(W, 0, None)

    return xr.Dataset(
        {"temperature":  (["lat","lon","year","month"], T),
         "precipitation":(["lat","lon","year","month"], P),
         "wind_speed":   (["lat","lon","year","month"], W)},
        coords={"lat":lats,"lon":lons,"year":years,"month":months},
    )

@st.cache_data
def get_spatial_slice(variable, year, month):
    ds  = generate_synthetic_dataset()
    arr = ds[variable].sel(year=year, month=month).values
    lats, lons = ds.lat.values, ds.lon.values
    lg, nog = np.meshgrid(lats, lons, indexing='ij')
    return pd.DataFrame({"lat":lg.ravel(),"lon":nog.ravel(),"value":arr.ravel()})

@st.cache_data
def get_time_series(lat, lon, variable, start_year):
    ds  = generate_synthetic_dataset()
    near= ds.sel(lat=lat, lon=lon, method="nearest")
    rows=[]
    for y in ds.year.values:
        if y < start_year: continue
        for m in ds.month.values:
            rows.append({"date":pd.Timestamp(year=int(y),month=int(m),day=15),
                         "value":float(near[variable].sel(year=y,month=m).values),
                         "year":int(y),"month":int(m)})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    x  = np.arange(len(df))
    c  = np.polyfit(x, df["value"], 1)
    df["trend"] = np.polyval(c, x)
    df["trend_slope_per_decade"] = c[0]*120
    return df

@st.cache_data
def get_annual_mean(variable, year):
    ds  = generate_synthetic_dataset()
    arr = ds[variable].sel(year=year).mean(dim="month").values
    lats,lons = ds.lat.values, ds.lon.values
    lg,nog = np.meshgrid(lats,lons,indexing='ij')
    return pd.DataFrame({"lat":lg.ravel(),"lon":nog.ravel(),"value":arr.ravel()})

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,18,35,0.55)",
    font=dict(family="Space Grotesk, sans-serif", color="#94a3b8"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=40,r=20,t=40,b=40),
)
CMAPS   = {"Thermal":"RdBu_r","Viridis":"Viridis","Plasma":"Plasma","Ocean":"Blues","Inferno":"Inferno"}
VAR_META= {
    "temperature":  {"label":"Surface Temperature","unit":"°C",    "icon":"🌡️","cmap":"Thermal"},
    "precipitation":{"label":"Precipitation",       "unit":"mm/day","icon":"🌧️","cmap":"Viridis"},
    "wind_speed":   {"label":"Wind Speed",           "unit":"m/s",  "icon":"💨","cmap":"Plasma"},
}


#  SIDEBAR — SINGLE GLOBAL LOCATION SELECTOR
#  Everything below reads from: g_lat, g_lon, g_loc_name

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:16px;">
      <div style="font-size:22px;font-weight:700;color:#00d4aa;text-shadow:0 0 16px rgba(0,212,170,0.5);">
        🌍 PyClimaExplorer
      </div>
      <div style="font-size:11px;color:#64748b;letter-spacing:1px;margin-top:4px;">TECHNEX '26</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">🔍 Location — drives all tabs</div>', unsafe_allow_html=True)

    city_list  = list(WORLD_CITIES.keys())
    chosen_key = st.selectbox("Select location", city_list, key="g_city")
    coords     = WORLD_CITIES[chosen_key]
    def_lat    = float(coords[0]) if coords else 25.3
    def_lon    = float(coords[1]) if coords else 83.0

    # When the city changes, overwrite the slider widget keys directly.
    # Writing to the widget key BEFORE the slider is rendered forces Streamlit
    # to use the new value instead of its cached internal state.
    if st.session_state.get("_g_last_city") != chosen_key:
        st.session_state["_g_last_city"] = chosen_key
        st.session_state["g_lat_sl"]     = def_lat   # ← widget key, not a shadow key
        st.session_state["g_lon_sl"]     = def_lon

    g_lat = st.slider("Latitude  (N+, S−)", -90.0, 90.0, step=2.5, key="g_lat_sl")
    g_lon = st.slider("Longitude (E+, W−)", -180.0, 180.0, step=2.5, key="g_lon_sl")

    lat_dir = "N" if g_lat >= 0 else "S"
    lon_dir = "E" if g_lon >= 0 else "W"
    g_loc_name = chosen_key.split(",")[0].translate(EMOJI_STRIP).strip()
    if "Custom" in chosen_key or "coordinates" in chosen_key:
        g_loc_name = f"{abs(g_lat):.1f}°{lat_dir}, {abs(g_lon):.1f}°{lon_dir}"

    st.markdown(
        f'<div class="loc-badge">'
        f'<div class="loc-name">{g_loc_name}</div>'
        f'<div class="loc-coords">{abs(g_lat):.1f}°{lat_dir}, {abs(g_lon):.1f}°{lon_dir}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">🎛️ Global Controls</div>', unsafe_allow_html=True)
    g_var   = st.selectbox("Variable",
                           list(VAR_META.keys()),
                           format_func=lambda k: f"{VAR_META[k]['icon']} {VAR_META[k]['label']}",
                           key="g_var")
    g_year  = st.slider("Year",  1950, 2024, 2020, key="g_year")
    g_month = st.selectbox("Month", list(range(1,13)), index=6,
                           format_func=lambda m: MONTH_NAMES[m-1], key="g_month")
    g_cmap  = st.selectbox("Color Palette", list(CMAPS.keys()), key="g_cmap")

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">ℹ️ Dataset</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:11px;color:#64748b;line-height:1.9;">
    <b style="color:#94a3b8">Format:</b> NetCDF (.nc)<br>
    <b style="color:#94a3b8">Resolution:</b> 5° × 5°<br>
    <b style="color:#94a3b8">Period:</b> 1950 – 2024<br>
    <b style="color:#94a3b8">Source:</b> ERA5 / CESM synthetic<br>
    </div>""", unsafe_allow_html=True)



#  HEADER

st.markdown(f"""
<div class="main-header">
  <div>
    <div class="header-logo">🌍 PyClimaExplorer</div>
    <div style="font-size:12px;color:#64748b;margin-top:4px;">
      Showing all data for &nbsp;<b style="color:#00d4aa">{g_loc_name}</b>
      &nbsp;·&nbsp; {abs(g_lat):.1f}°{lat_dir}, {abs(g_lon):.1f}°{lon_dir}
      &nbsp;·&nbsp; {MONTH_NAMES[g_month-1]} {g_year}
    </div>
  </div>
  <div>
    <span class="header-badge">TECHNEX '26</span>
    &nbsp;<span style="font-size:11px;color:#64748b;">Python · Streamlit · Plotly · Xarray</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Fetch shared data 
df_spatial = get_spatial_slice(g_var, g_year, g_month)
df_ts_full = get_time_series(g_lat, g_lon, g_var, 1950)
meta       = VAR_META[g_var]
colorscale = CMAPS[g_cmap]
mean_v     = df_spatial["value"].mean()
std_v      = df_spatial["value"].std()

# Location-specific values
loc_val    = float(generate_synthetic_dataset()[g_var].sel(
    lat=g_lat, lon=g_lon, method="nearest").sel(year=g_year, month=g_month).values)
slope_dec  = df_ts_full["trend_slope_per_decade"].iloc[0]
loc_mean   = df_ts_full["value"].mean()
loc_anom   = loc_val - loc_mean

# KPI row (location-aware) 
k1,k2,k3,k4 = st.columns(4)
with k1:
    st.markdown(
        f'<div class="metric-card accent-teal">'
        f'<div class="metric-label">{g_loc_name} — {MONTH_NAMES[g_month-1]} {g_year}</div>'
        f'<div class="metric-value">{loc_val:.1f}{meta["unit"]}</div>'
        f'<div class="metric-sub">At selected location</div></div>',
        unsafe_allow_html=True)
with k2:
    anom_col = "accent-red" if loc_anom > 0 else "accent-blue"
    sign     = "+" if loc_anom > 0 else ""
    st.markdown(
        f'<div class="metric-card {anom_col}">'
        f'<div class="metric-label">Anomaly vs 1950–2024 mean</div>'
        f'<div class="metric-value">{sign}{loc_anom:.2f}{meta["unit"]}</div>'
        f'<div class="metric-sub">Departure from long-term avg</div></div>',
        unsafe_allow_html=True)
with k3:
    t_col = "accent-red" if slope_dec > 0 else "accent-blue"
    sign  = "+" if slope_dec > 0 else ""
    st.markdown(
        f'<div class="metric-card {t_col}">'
        f'<div class="metric-label">Local trend / decade</div>'
        f'<div class="metric-value">{sign}{slope_dec:.2f}{meta["unit"]}</div>'
        f'<div class="metric-sub">Linear regression 1950–2024</div></div>',
        unsafe_allow_html=True)
with k4:
    zscore = (loc_val - mean_v) / std_v if std_v > 0 else 0
    level  = "EXTREME" if abs(zscore)>3 else ("HIGH" if abs(zscore)>2 else ("MOD" if abs(zscore)>1 else "LOW"))
    lc     = {"EXTREME":"#ef4444","HIGH":"#f59e0b","MOD":"#eab308","LOW":"#00d4aa"}[level]
    st.markdown(
        f'<div class="metric-card accent-orange">'
        f'<div class="metric-label">Anomaly level (global grid)</div>'
        f'<div class="metric-value" style="color:{lc}">{level}</div>'
        f'<div class="metric-sub">Z-score: {zscore:+.2f}σ</div></div>',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


#  TABS

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13 = st.tabs([
    "🗺️  Spatial Heatmap",
    "📈  Time-Series",
    "🌐  3D Globe",
    "⚖️  Compare Mode",
    "📖  Guided Story",
    "🤖  AI Assistant",
    "🔮  Climate Predictions",
    "⚠️  Extreme Events",
    "🧠  Climate Insights",
    "🎬  Climate Animation",
    "🖱️  Interactive Analysis",
    "🌍  Climate Simulator",
    "🔬  Climate Analytics",
])


#  TAB 1 — SPATIAL HEATMAP  (centred on g_lat, g_lon)

with tab1:
    st.markdown(f"#### 🗺️ Global Heatmap — {meta['label']} · {MONTH_NAMES[g_month-1]} {g_year} · pin = {g_loc_name}")

    sigma = st.slider("Anomaly threshold σ", 1.0, 3.0, 2.0, 0.5, key="t1_sigma")
    df_sp = df_spatial.copy()
    df_sp["z"] = (df_sp["value"] - mean_v) / std_v

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattergeo(
        lat=df_sp["lat"], lon=df_sp["lon"], mode="markers",
        marker=dict(size=5, color=df_sp["value"], colorscale=colorscale, showscale=True,
                    colorbar=dict(title=dict(text=f"{meta['label']} ({meta['unit']})",
                                             font=dict(color="#94a3b8",size=11)),
                                  tickfont=dict(color="#94a3b8",size=10), len=0.7, x=1.01),
                    opacity=0.82),
        text=[f"<b>{la:.1f}°{'N' if la>=0 else 'S'}, {lo:.1f}°{'E' if lo>=0 else 'W'}</b><br>"
              f"{meta['label']}: <b>{v:.2f} {meta['unit']}</b><br>Z: {z:.2f}σ"
              for la,lo,v,z in zip(df_sp["lat"],df_sp["lon"],df_sp["value"],df_sp["z"])],
        hoverinfo="text", name=meta['label'],
    ))
    hot  = df_sp[df_sp["z"] >  sigma]
    cold = df_sp[df_sp["z"] < -sigma]
    if len(hot):
        fig_map.add_trace(go.Scattergeo(lat=hot["lat"],lon=hot["lon"],mode="markers",
            marker=dict(size=9,color="rgba(239,68,68,0.5)",line=dict(width=1,color="rgba(239,68,68,0.8)")),
            name=f"Hot >{sigma}σ", hoverinfo="skip"))
    if len(cold):
        fig_map.add_trace(go.Scattergeo(lat=cold["lat"],lon=cold["lon"],mode="markers",
            marker=dict(size=9,color="rgba(34,211,238,0.4)",line=dict(width=1,color="rgba(34,211,238,0.7)")),
            name=f"Cold <-{sigma}σ", hoverinfo="skip"))
    # PIN — selected location
    fig_map.add_trace(go.Scattergeo(
        lat=[g_lat], lon=[g_lon], mode="markers+text",
        marker=dict(size=16, color="#00d4aa", symbol="star",
                    line=dict(width=2, color="white")),
        text=[f"  {g_loc_name}"], textfont=dict(color="#00d4aa", size=12),
        textposition="middle right", name="Selected location",
        hovertext=f"<b>{g_loc_name}</b><br>{loc_val:.2f} {meta['unit']}",
        hoverinfo="text",
    ))

    fig_map.update_geos(
        showland=True, landcolor="rgba(20,35,55,0.85)",
        showocean=True, oceancolor="rgba(5,15,35,0.80)",
        showlakes=True, lakecolor="rgba(5,15,35,0.75)",
        showcoastlines=True, coastlinecolor="rgba(100,116,139,0.4)", coastlinewidth=0.5,
        showframe=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)",
        # Centre map on selected location
        center=dict(lat=g_lat, lon=g_lon),
    )
    fig_map.update_layout(
        title=dict(text=f"<b>{meta['label']}</b> — {MONTH_NAMES[g_month-1]} {g_year}  ⭐ = {g_loc_name}",
                   font=dict(size=15,color="#e2e8f0"),x=0.01),
        paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)",
        height=520, margin=dict(l=0,r=0,t=44,b=0),
        legend=dict(font=dict(color="#94a3b8",size=11),
                    bgcolor="rgba(10,18,35,0.7)",bordercolor="rgba(255,255,255,0.1)",borderwidth=1),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Global Mean",    f"{mean_v:.2f} {meta['unit']}")
    c2.metric(f"{g_loc_name} value", f"{loc_val:.2f} {meta['unit']}", delta=f"{loc_anom:+.2f}")
    c3.metric("Anomaly cells",  f"{len(hot)+len(cold)} / {len(df_sp)}", delta=f"threshold ±{sigma}σ")



#  TAB 2 — TIME-SERIES  (for g_lat, g_lon)

with tab2:
    st.markdown(f"#### 📈 Time-Series — {meta['label']} at {g_loc_name}")

    tc1,tc2 = st.columns(2)
    with tc1:
        ts_range = st.selectbox("Date Range",[5,10,20,50,74],index=2,
                                format_func=lambda y: f"Last {y} years" if y<74 else "Full (1950–2024)")
        ts_agg   = st.selectbox("Aggregation",["Monthly","Annual Mean"],key="t2_agg")
    with tc2:
        show_trend   = st.checkbox("Show trend line",  True, key="t2_trend")
        show_anomaly = st.checkbox("Highlight anomalies (>2σ)", True, key="t2_anom")
        chart_type   = st.radio("Chart type",["Line","Bar"],horizontal=True,key="t2_chart")

    start_year = 2024 - ts_range
    df_ts = get_time_series(g_lat, g_lon, g_var, start_year)

    if ts_agg == "Annual Mean":
        df_plot = df_ts.groupby("year")["value"].mean().reset_index()
        df_plot["date"]  = pd.to_datetime(df_plot["year"].astype(str))
        df_plot["trend"] = np.polyval(np.polyfit(np.arange(len(df_plot)),df_plot["value"],1),np.arange(len(df_plot)))
    else:
        df_plot = df_ts.copy()
    x_col = "date"

    mean_ts  = df_plot["value"].mean()
    std_ts   = df_plot["value"].std()
    slope    = df_ts["trend_slope_per_decade"].iloc[0]

    mc1,mc2,mc3,mc4 = st.columns(4)
    with mc1:
        st.markdown(f'<div class="metric-card accent-blue"><div class="metric-label">Location</div>'
                    f'<div class="metric-value" style="font-size:14px;padding-top:6px;">{g_loc_name}</div>'
                    f'<div class="metric-sub">{abs(g_lat):.1f}°{lat_dir}, {abs(g_lon):.1f}°{lon_dir}</div></div>',
                    unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="metric-card accent-teal"><div class="metric-label">Period Mean</div>'
                    f'<div class="metric-value">{mean_ts:.1f}{meta["unit"]}</div>'
                    f'<div class="metric-sub">{start_year}–2024</div></div>', unsafe_allow_html=True)
    with mc3:
        tc = "accent-red" if slope>0 else "accent-blue"
        st.markdown(f'<div class="metric-card {tc}"><div class="metric-label">Trend / Decade</div>'
                    f'<div class="metric-value">{"+" if slope>0 else ""}{slope:.2f}{meta["unit"]}</div>'
                    f'<div class="metric-sub">Linear regression</div></div>', unsafe_allow_html=True)
    with mc4:
        rec = df_plot["value"].max()
        rd  = df_plot.loc[df_plot["value"].idxmax(),"date"]
        rds = rd.strftime("%b %Y") if hasattr(rd,"strftime") else str(rd)
        st.markdown(f'<div class="metric-card accent-orange"><div class="metric-label">Record High</div>'
                    f'<div class="metric-value">{rec:.1f}{meta["unit"]}</div>'
                    f'<div class="metric-sub">{rds}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig_ts = go.Figure()
    if chart_type == "Bar":
        cols = ["rgba(239,68,68,0.7)" if v>mean_ts else "rgba(34,211,238,0.6)" for v in df_plot["value"]]
        fig_ts.add_trace(go.Bar(x=df_plot[x_col],y=df_plot["value"],marker_color=cols,name=g_loc_name,
            hovertemplate=f"%{{x|%b %Y}}<br><b>%{{y:.2f}} {meta['unit']}</b><extra></extra>"))
    else:
        if show_anomaly:
            hm = df_plot["value"] > mean_ts+2*std_ts
            cm = df_plot["value"] < mean_ts-2*std_ts
            if hm.any():
                fig_ts.add_trace(go.Scatter(x=df_plot.loc[hm,x_col],y=df_plot.loc[hm,"value"],
                    mode="markers",marker=dict(size=6,color="rgba(239,68,68,0.8)"),name="Hot >2σ",hoverinfo="skip"))
            if cm.any():
                fig_ts.add_trace(go.Scatter(x=df_plot.loc[cm,x_col],y=df_plot.loc[cm,"value"],
                    mode="markers",marker=dict(size=6,color="rgba(34,211,238,0.8)"),name="Cold <-2σ",hoverinfo="skip"))
        fig_ts.add_trace(go.Scatter(x=df_plot[x_col],y=df_plot["value"],mode="lines",name=g_loc_name,
            line=dict(color="#3b82f6",width=1.8),fill="tozeroy",fillcolor="rgba(59,130,246,0.07)",
            hovertemplate=f"%{{x|%b %Y}}<br><b>%{{y:.2f}} {meta['unit']}</b><extra></extra>"))

    if show_trend and "trend" in df_plot.columns:
        fig_ts.add_trace(go.Scatter(x=df_plot[x_col],y=df_plot["trend"],mode="lines",
            name="Trend",line=dict(color="#ef4444",width=2,dash="dot"),hoverinfo="skip"))

    # Mark current selected year/month
    sel_date = pd.Timestamp(year=g_year, month=g_month, day=15)
    if df_plot[x_col].min() <= sel_date <= df_plot[x_col].max():
        vline(fig_ts, sel_date.strftime("%Y-%m-%d"),
              color="rgba(0,212,170,0.6)", dash="dash", width=2,
              label=f"{MONTH_NAMES[g_month-1]} {g_year}", label_color="#00d4aa")

    hline(fig_ts, mean_ts, color="rgba(245,158,11,0.4)", dash="dash", width=1,
          label=f"Mean {mean_ts:.1f}{meta['unit']}", label_color="#f59e0b")
    fig_ts.update_layout(**DARK,
        title=dict(text=f"<b>{meta['label']}</b> — {g_loc_name}  ({start_year}–2024)",
                   font=dict(size=14,color="#e2e8f0"),x=0.01),
        height=360,showlegend=True,
        legend=dict(font=dict(color="#94a3b8",size=11),bgcolor="rgba(10,18,35,0.7)",
                    bordercolor="rgba(255,255,255,0.1)",borderwidth=1),
        yaxis_title=f"{meta['label']} ({meta['unit']})",
        yaxis_title_font=dict(color="#64748b",size=11),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("##### Seasonal Climatology — 12-Month Cycle")
    seas = df_ts.groupby("month")["value"].mean().reset_index()
    seas["name"]  = MONTH_NAMES
    seas["color"] = ["rgba(239,68,68,0.7)" if v>=mean_ts else "rgba(34,211,238,0.6)" for v in seas["value"]]
    fig_s = go.Figure(go.Bar(x=seas["name"],y=seas["value"],marker_color=seas["color"],
        hovertemplate="%{x}<br><b>%{y:.2f} "+meta["unit"]+"</b><extra></extra>"))
    # Highlight current month
    vline(fig_s, MONTH_NAMES[g_month-1], color="rgba(0,212,170,0.7)", dash="dot", width=2)
    fig_s.update_layout(**DARK,height=230,showlegend=False,yaxis_title=meta["unit"],
        title=dict(text=f"12-month climatology — {g_loc_name}",font=dict(size=13,color="#94a3b8"),x=0.01))
    st.plotly_chart(fig_s, use_container_width=True)



#  TAB 3 — 3D GLOBE  (selected location highlighted)

with tab3:
    st.markdown(f"#### 🌐 3D Globe — {meta['label']} · {MONTH_NAMES[g_month-1]} {g_year} · ⭐ {g_loc_name}")

    gc1, gc2 = st.columns([1, 3])
    with gc1:
        globe_proj = st.selectbox("Projection",["orthographic","natural earth","equirectangular","mollweide"],key="t3_proj")
        show_anomaly_globe = st.checkbox("Show anomaly vs 1980", True, key="t3_anom")

        st.markdown("---")
        st.markdown(f"""<div class="metric-card accent-teal" style="margin-bottom:10px;">
        <div class="metric-label">{g_loc_name}</div>
        <div class="metric-value" style="font-size:20px;">{loc_val:.1f}{meta["unit"]}</div>
        <div class="metric-sub">{MONTH_NAMES[g_month-1]} {g_year}</div>
        </div>""", unsafe_allow_html=True)
        anom_str = f"{loc_anom:+.2f}{meta['unit']}"
        ac = "accent-red" if loc_anom>0 else "accent-blue"
        st.markdown(f"""<div class="metric-card {ac}" style="margin-bottom:10px;">
        <div class="metric-label">Local anomaly</div>
        <div class="metric-value" style="font-size:20px;">{anom_str}</div>
        <div class="metric-sub">vs 1950–2024 mean</div>
        </div>""", unsafe_allow_html=True)

    with gc2:
        df_globe = df_spatial.copy()
        if show_anomaly_globe:
            df_base = get_spatial_slice(g_var, 1980, g_month)
            df_globe["plot_val"] = df_globe["value"].values - df_base["value"].values
            cbar_title = f"Anomaly vs 1980 ({meta['unit']})"
            cscale, cmin, cmax = "RdBu_r", -4, 4
        else:
            df_globe["plot_val"] = df_globe["value"]
            cbar_title = f"{meta['label']} ({meta['unit']})"
            cscale, cmin, cmax = colorscale, df_globe["plot_val"].min(), df_globe["plot_val"].max()

        fig_g = go.Figure()
        fig_g.add_trace(go.Scattergeo(
            lat=df_globe["lat"], lon=df_globe["lon"], mode="markers",
            marker=dict(size=6, color=df_globe["plot_val"], colorscale=cscale,
                        cmin=cmin, cmax=cmax, showscale=True, opacity=0.88,
                        colorbar=dict(title=dict(text=cbar_title,font=dict(color="#94a3b8",size=11)),
                                      tickfont=dict(color="#94a3b8",size=10),len=0.6)),
            text=[f"<b>{la:.1f}°{'N' if la>=0 else 'S'}, {lo:.1f}°{'E' if lo>=0 else 'W'}</b><br>"
                  f"Value: <b>{v:+.2f} {meta['unit']}</b>"
                  for la,lo,v in zip(df_globe["lat"],df_globe["lon"],df_globe["plot_val"])],
            hoverinfo="text", name="Grid",
        ))
        # PIN
        pin_val = df_globe.loc[(df_globe["lat"]-g_lat).abs().idxmin(), "plot_val"] if show_anomaly_globe else loc_val
        fig_g.add_trace(go.Scattergeo(
            lat=[g_lat], lon=[g_lon], mode="markers+text",
            marker=dict(size=18, color="#00d4aa", symbol="star", line=dict(width=2,color="white")),
            text=[f"  {g_loc_name}"], textfont=dict(color="#00d4aa",size=11),
            textposition="middle right", name="Selected",
            hovertext=f"<b>{g_loc_name}</b><br>{pin_val:+.2f} {meta['unit']}",
            hoverinfo="text",
        ))
        fig_g.update_geos(
            projection_type=globe_proj,
            showland=True, landcolor="rgba(20,35,55,0.95)",
            showocean=True, oceancolor="rgba(5,15,40,0.95)",
            showlakes=True, lakecolor="rgba(5,15,40,0.8)",
            showcoastlines=True, coastlinecolor="rgba(100,116,139,0.5)", coastlinewidth=0.6,
            showcountries=True, countrycolor="rgba(100,116,139,0.2)",
            showframe=False, bgcolor="rgba(5,13,26,1)",
            # Point globe at selected location
            projection_rotation=dict(lon=g_lon, lat=g_lat, roll=0),
        )
        fig_g.update_layout(
            paper_bgcolor="rgba(5,13,26,0.8)", height=500, margin=dict(l=0,r=0,t=40,b=0),
            title=dict(text=f"<b>{meta['label']}</b> — {MONTH_NAMES[g_month-1]} {g_year}  ⭐ = {g_loc_name}",
                       font=dict(size=14,color="#e2e8f0"),x=0.01),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    # Local long-term trend mini chart
    st.markdown(f"##### {g_loc_name} — Annual Mean {meta['label']} 1950–2024")
    ann = df_ts_full.groupby("year")["value"].mean().reset_index()
    fig_ann = go.Figure()
    fig_ann.add_trace(go.Scatter(x=ann["year"],y=ann["value"],mode="lines",name=g_loc_name,
        line=dict(color="#3b82f6",width=1.5),fill="tozeroy",fillcolor="rgba(59,130,246,0.07)"))
    fig_ann.add_trace(go.Scatter(x=ann["year"],
        y=np.polyval(np.polyfit(ann["year"],ann["value"],1),ann["year"]),
        mode="lines",name="Trend",line=dict(color="#ef4444",width=2,dash="dot")))
    vline(fig_ann, g_year, color="rgba(0,212,170,0.6)", dash="dash", width=2,
          label=str(g_year), label_color="#00d4aa")
    fig_ann.update_layout(**DARK,height=220,showlegend=True,
        legend=dict(font=dict(color="#94a3b8",size=11),bgcolor="rgba(10,18,35,0.7)"),
        yaxis_title=meta["unit"])
    st.plotly_chart(fig_ann, use_container_width=True)





#  TAB 4 — COMPARE MODE
#  Location A  = sidebar selection  (g_lat, g_lon)
#  Location B  = chosen inside this tab
#  Shows:
#    • Side-by-side global maps  (Year A | Year B)
#      with BOTH location pins on EVERY map
#    • Monthly profile: A vs B for the selected year
#    • Annual time-series: A vs B from 1990–2024
#    • Zonal mean profile with both latitude lines
#    • Diff stat cards (global + local A vs local B)

with tab4:
    st.markdown(f"#### ⚖️ Compare Mode — {g_loc_name} vs a Second Location")

    # Controls 
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    with cc1:
        cmp_search = st.text_input(
            "🔍 Search Location B",
            placeholder="e.g. London, Arctic, Tokyo…",
            key="t4_search"
        )
        cmp_filt = (
            {k: v for k, v in WORLD_CITIES.items() if cmp_search.lower() in k.lower()}
            if cmp_search else WORLD_CITIES
        )
        if not cmp_filt:
            cmp_filt = WORLD_CITIES
        cmp_city   = st.selectbox(f"Location B  ({len(cmp_filt)} shown)", list(cmp_filt.keys()), key="t4_city")
        cmp_coords = cmp_filt[cmp_city]
        cmp_def_lat = float(cmp_coords[0]) if cmp_coords else 51.5
        cmp_def_lon = float(cmp_coords[1]) if cmp_coords else -0.1

        # Force slider widget keys when city changes
        if st.session_state.get("_t4_last_city") != cmp_city:
            st.session_state["_t4_last_city"] = cmp_city
            st.session_state["t4_lat"]         = cmp_def_lat
            st.session_state["t4_lon"]         = cmp_def_lon

        cmp_lat = st.slider("Latitude B",  -90.0,  90.0,  step=2.5, key="t4_lat")
        cmp_lon = st.slider("Longitude B", -180.0, 180.0, step=2.5, key="t4_lon")
        cmp_name = cmp_city.split(",")[0].translate(EMOJI_STRIP).strip()

    with cc2:
        cmp_y1 = st.selectbox("Year A (baseline)",   [1980, 1990, 2000, 2010], index=1, key="t4_y1")
        cmp_y2 = st.selectbox("Year B (comparison)", [2000, 2010, 2020, 2024], index=2, key="t4_y2")

    with cc3:
        st.markdown(
            f'<div class="metric-card accent-teal" style="margin-bottom:10px;">'
            f'<div class="metric-label">Location A (sidebar)</div>'
            f'<div class="metric-value" style="font-size:13px;padding-top:4px;">{g_loc_name}</div>'
            f'<div class="metric-sub">{abs(g_lat):.1f}°{lat_dir}, {abs(g_lon):.1f}°{lon_dir}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="metric-card accent-orange">'
            f'<div class="metric-label">Location B</div>'
            f'<div class="metric-value" style="font-size:13px;padding-top:4px;">{cmp_name}</div>'
            f'<div class="metric-sub">{abs(cmp_lat):.1f}°{"N" if cmp_lat>=0 else "S"}, '
            f'{abs(cmp_lon):.1f}°{"E" if cmp_lon>=0 else "W"}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Fetch annual-mean spatial slices
    df_c1 = get_annual_mean(g_var, cmp_y1)
    df_c2 = get_annual_mean(g_var, cmp_y2)
    shared_min = min(df_c1["value"].min(), df_c2["value"].min())
    shared_max = max(df_c1["value"].max(), df_c2["value"].max())

    #Fetch local values at both locations 
    ds_cmp   = generate_synthetic_dataset()
    val_A_y1 = float(ds_cmp[g_var].sel(lat=g_lat, lon=g_lon,   method="nearest").sel(year=cmp_y1).mean(dim="month").values)
    val_A_y2 = float(ds_cmp[g_var].sel(lat=g_lat, lon=g_lon,   method="nearest").sel(year=cmp_y2).mean(dim="month").values)
    val_B_y1 = float(ds_cmp[g_var].sel(lat=cmp_lat, lon=cmp_lon, method="nearest").sel(year=cmp_y1).mean(dim="month").values)
    val_B_y2 = float(ds_cmp[g_var].sel(lat=cmp_lat, lon=cmp_lon, method="nearest").sel(year=cmp_y2).mean(dim="month").values)
    diff_A   = val_A_y2 - val_A_y1
    diff_B   = val_B_y2 - val_B_y1

    #Side-by-side global maps with BOTH pins 
    def cmp_global_map(df, year, subtitle):
        fig = go.Figure()
        # Heatmap layer
        fig.add_trace(go.Scattergeo(
            lat=df["lat"], lon=df["lon"], mode="markers",
            marker=dict(size=4.5, color=df["value"], colorscale="RdBu_r",
                        cmin=shared_min, cmax=shared_max, showscale=False, opacity=0.8),
            hovertemplate=f"%{{lat:.1f}}°, %{{lon:.1f}}°<br>"
                          f"<b>%{{marker.color:.2f}} {meta['unit']}</b><extra></extra>",
            name=meta["label"],
        ))
        # Pin A — teal star
        val_A = val_A_y1 if year == cmp_y1 else val_A_y2
        fig.add_trace(go.Scattergeo(
            lat=[g_lat], lon=[g_lon], mode="markers+text",
            marker=dict(size=16, color="#00d4aa", symbol="star",
                        line=dict(width=2, color="white")),
            text=[f"  A: {val_A:.1f}{meta['unit']}"],
            textfont=dict(color="#00d4aa", size=11),
            textposition="middle right",
            name=f"A: {g_loc_name}",
            hovertemplate=f"<b>📍 A: {g_loc_name}</b><br>{year}: {val_A:.2f} {meta['unit']}<extra></extra>",
        ))
        # Pin B — orange star
        val_B = val_B_y1 if year == cmp_y1 else val_B_y2
        fig.add_trace(go.Scattergeo(
            lat=[cmp_lat], lon=[cmp_lon], mode="markers+text",
            marker=dict(size=16, color="#f59e0b", symbol="star",
                        line=dict(width=2, color="white")),
            text=[f"  B: {val_B:.1f}{meta['unit']}"],
            textfont=dict(color="#f59e0b", size=11),
            textposition="middle right",
            name=f"B: {cmp_name}",
            hovertemplate=f"<b>📍 B: {cmp_name}</b><br>{year}: {val_B:.2f} {meta['unit']}<extra></extra>",
        ))
        fig.update_geos(
            projection_type="natural earth",
            showland=True,  landcolor="rgba(20,35,55,0.9)",
            showocean=True, oceancolor="rgba(5,15,35,0.9)",
            showcoastlines=True, coastlinecolor="rgba(100,116,139,0.4)", coastlinewidth=0.5,
            showframe=False, bgcolor="rgba(0,0,0,0)",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(l=0, r=0, t=36, b=0),
            title=dict(
                text=f"<b>{year}</b> — {subtitle}  |  ⭐ A={g_loc_name}  🌟 B={cmp_name}",
                font=dict(size=12, color="#e2e8f0"), x=0.5, xanchor="center"
            ),
            legend=dict(font=dict(color="#94a3b8", size=10),
                        bgcolor="rgba(10,18,35,0.7)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        )
        return fig

    mc1, mc2 = st.columns(2)
    with mc1:
        st.plotly_chart(cmp_global_map(df_c1, cmp_y1, "Baseline"), use_container_width=True)
    with mc2:
        st.plotly_chart(cmp_global_map(df_c2, cmp_y2, "Comparison"), use_container_width=True)

    # Diff stat cards 
    global_diff   = df_c2["value"].mean() - df_c1["value"].mean()
    global_pct    = (global_diff / abs(df_c1["value"].mean())) * 100
    warm_cells    = ((df_c2["value"].values - df_c1["value"].values) > 0).mean() * 100
    diff_A_col    = "#ef4444" if diff_A > 0 else "#22d3ee"
    diff_B_col    = "#ef4444" if diff_B > 0 else "#22d3ee"

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        sg = "+" if global_diff >= 0 else ""
        st.markdown(f'<div class="diff-banner"><div class="diff-num">{sg}{global_diff:.2f}{meta["unit"]}</div>'
                    f'<div class="diff-label">Global mean change<br>{cmp_y1} → {cmp_y2}</div></div>',
                    unsafe_allow_html=True)
    with d2:
        sg = "+" if diff_A >= 0 else ""
        st.markdown(f'<div class="diff-banner"><div class="diff-num" style="color:{diff_A_col};">'
                    f'{sg}{diff_A:.2f}{meta["unit"]}</div>'
                    f'<div class="diff-label">📍 A: {g_loc_name}<br>{cmp_y1} → {cmp_y2}</div></div>',
                    unsafe_allow_html=True)
    with d3:
        sg = "+" if diff_B >= 0 else ""
        st.markdown(f'<div class="diff-banner"><div class="diff-num" style="color:{diff_B_col};">'
                    f'{sg}{diff_B:.2f}{meta["unit"]}</div>'
                    f'<div class="diff-label">📍 B: {cmp_name}<br>{cmp_y1} → {cmp_y2}</div></div>',
                    unsafe_allow_html=True)
    with d4:
        ab_diff = val_A_y2 - val_B_y2
        ab_col  = "#ef4444" if ab_diff > 0 else "#22d3ee"
        sg = "+" if ab_diff >= 0 else ""
        st.markdown(f'<div class="diff-banner"><div class="diff-num" style="color:{ab_col};">'
                    f'{sg}{ab_diff:.2f}{meta["unit"]}</div>'
                    f'<div class="diff-label">A vs B in {cmp_y2}<br>({g_loc_name} − {cmp_name})</div></div>',
                    unsafe_allow_html=True)
    with d5:
        sg = "+" if global_pct >= 0 else ""
        st.markdown(f'<div class="diff-banner"><div class="diff-num" style="color:#22d3ee;">'
                    f'{sg}{global_pct:.1f}%</div>'
                    f'<div class="diff-label">Global % change<br>{cmp_y1} → {cmp_y2}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly profile chart  (selected year, both locations)
    st.markdown(f"##### Monthly Profile in {cmp_y2} — Location A vs B")
    ds_mp  = generate_synthetic_dataset()
    mo_A   = [float(ds_mp[g_var].sel(lat=g_lat,   lon=g_lon,   method="nearest").sel(year=cmp_y2, month=m).values) for m in range(1, 13)]
    mo_B   = [float(ds_mp[g_var].sel(lat=cmp_lat, lon=cmp_lon, method="nearest").sel(year=cmp_y2, month=m).values) for m in range(1, 13)]
    fig_mo = go.Figure()
    fig_mo.add_trace(go.Scatter(
        x=MONTH_NAMES, y=mo_A, mode="lines+markers", name=f"A: {g_loc_name}",
        line=dict(color="#00d4aa", width=2.5), marker=dict(size=8, color="#00d4aa"),
        hovertemplate=f"%{{x}}<br><b>%{{y:.2f}} {meta['unit']}</b>  ({g_loc_name})<extra></extra>",
    ))
    fig_mo.add_trace(go.Scatter(
        x=MONTH_NAMES, y=mo_B, mode="lines+markers", name=f"B: {cmp_name}",
        line=dict(color="#f59e0b", width=2.5), marker=dict(size=8, color="#f59e0b"),
        hovertemplate=f"%{{x}}<br><b>%{{y:.2f}} {meta['unit']}</b>  ({cmp_name})<extra></extra>",
    ))
    # vertical line on currently selected month
    vline(fig_mo, MONTH_NAMES[g_month-1], color="rgba(255,255,255,0.3)", dash="dot", width=1,
          label=MONTH_NAMES[g_month-1], label_color="#94a3b8")
    fig_mo.update_layout(**DARK, height=280, showlegend=True,
        title=dict(text=f"Monthly profile in {cmp_y2}  |  A = {g_loc_name}  vs  B = {cmp_name}",
                   font=dict(size=13, color="#e2e8f0"), x=0.01),
        legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(10,18,35,0.7)"),
        xaxis_title="Month", yaxis_title=f"{meta['label']} ({meta['unit']})")
    st.plotly_chart(fig_mo, use_container_width=True)

    # ── Annual time-series overlay  (Location A vs B, 1990–2024) ─
    st.markdown(f"##### Annual Mean {meta['label']} — {g_loc_name} vs {cmp_name}  (1990–2024)")
    df_ts_A = get_time_series(g_lat,   g_lon,   g_var, 1990)
    df_ts_B = get_time_series(cmp_lat, cmp_lon, g_var, 1990)
    ann_A   = df_ts_A.groupby("year")["value"].mean().reset_index()
    ann_B   = df_ts_B.groupby("year")["value"].mean().reset_index()

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=ann_A["year"], y=ann_A["value"], mode="lines", name=f"A: {g_loc_name}",
        line=dict(color="#00d4aa", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
        hovertemplate=f"%{{x}}<br><b>%{{y:.2f}} {meta['unit']}</b>  ({g_loc_name})<extra></extra>",
    ))
    fig_cmp.add_trace(go.Scatter(
        x=ann_B["year"], y=ann_B["value"], mode="lines", name=f"B: {cmp_name}",
        line=dict(color="#f59e0b", width=2.5),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.06)",
        hovertemplate=f"%{{x}}<br><b>%{{y:.2f}} {meta['unit']}</b>  ({cmp_name})<extra></extra>",
    ))
    # trend lines
    xa = ann_A["year"].values; ya = ann_A["value"].values
    xb = ann_B["year"].values; yb = ann_B["value"].values
    fig_cmp.add_trace(go.Scatter(x=xa, y=np.polyval(np.polyfit(xa,ya,1),xa),
        mode="lines", name=f"A trend", line=dict(color="#00d4aa",width=1.5,dash="dot"), hoverinfo="skip"))
    fig_cmp.add_trace(go.Scatter(x=xb, y=np.polyval(np.polyfit(xb,yb,1),xb),
        mode="lines", name=f"B trend", line=dict(color="#f59e0b",width=1.5,dash="dot"), hoverinfo="skip"))
    # mark selected year
    vline(fig_cmp, g_year, color="rgba(255,255,255,0.25)", dash="dash", width=1,
          label=str(g_year), label_color="#94a3b8")
    fig_cmp.update_layout(**DARK, height=300, showlegend=True,
        title=dict(text=f"Annual mean {meta['label']}  |  A = {g_loc_name}  vs  B = {cmp_name}",
                   font=dict(size=13, color="#e2e8f0"), x=0.01),
        legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(10,18,35,0.7)"),
        xaxis_title="Year", yaxis_title=f"{meta['label']} ({meta['unit']})")
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Zonal mean  (latitude profile with both location markers) ─
    st.markdown("##### Zonal Mean Profile — both latitude bands marked")
    ds_z   = generate_synthetic_dataset()
    lats_z = ds_z.lat.values
    z1 = ds_z[g_var].sel(year=cmp_y1).mean(dim=["lon","month"]).values
    z2 = ds_z[g_var].sel(year=cmp_y2).mean(dim=["lon","month"]).values
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(x=lats_z, y=z1, mode="lines", name=str(cmp_y1),
        line=dict(color="#3b82f6",width=2), fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"))
    fig_z.add_trace(go.Scatter(x=lats_z, y=z2, mode="lines", name=str(cmp_y2),
        line=dict(color="#ef4444",width=2), fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"))
    vline(fig_z, g_lat,  color="rgba(0,212,170,0.8)",  dash="dash", width=2,
          label=f"A: {g_loc_name}", label_color="#00d4aa")
    vline(fig_z, cmp_lat, color="rgba(245,158,11,0.8)", dash="dash", width=2,
          label=f"B: {cmp_name}",  label_color="#f59e0b")
    fig_z.update_layout(**DARK, height=260,
        xaxis_title="Latitude (°)", yaxis_title=meta["unit"],
        legend=dict(font=dict(color="#94a3b8",size=11), bgcolor="rgba(10,18,35,0.7)"))
    st.plotly_chart(fig_z, use_container_width=True)



#  TAB 5 — GUIDED STORY  (anomaly chart highlights selected location)

with tab5:
    st.markdown(f"#### 📖 Guided Climate Story — centred on {g_loc_name}")
    st.markdown(f'<div style="color:#64748b;font-size:13px;margin-bottom:18px;">'
                f'All charts in this tab use <b style="color:#00d4aa">{g_loc_name}</b> as the reference location. '
                f'Change location in the sidebar to see how the story shifts.</div>', unsafe_allow_html=True)

    STORIES = [
        {"year":"1950s–1970s","step":1950,"title":"The Baseline Era",
         "body":"The 1950–1980 period is the standard climate baseline for anomaly calculations. "
                "Arctic sea ice was at its highest, and extreme heat outside the tropics was rare. "
                "At your selected location, temperatures were near the long-term average.",
         "chips":[("Pre-industrial baseline","chip-cold"),("Normal variability","chip-cold")],
         "anomaly":None},
        {"year":"1983","step":1983,"title":"El Niño Disruption",
         "body":"The 1982–83 El Niño caused severe droughts across Australia, India, and Africa "
                "while triggering floods in the Americas. Global mean temperature spiked 0.28°C above baseline. "
                "The impact at your location depends strongly on its latitude and proximity to teleconnection centres.",
         "chips":[("El Niño event","chip-hot"),("Drought + floods","chip-hot")],
         "anomaly":("🌊 ENSO Anomaly — 1983",
                    "Eastern Pacific SST rose >3°C above normal. Teleconnection patterns reshaped precipitation and temperature anomalies worldwide — visible in your location's 1983 time-series spike above.")},
        {"year":"1998","step":1998,"title":"Super El Niño & Record Heat",
         "body":"1997–98 El Niño became the strongest on record. 1998 was the hottest year yet measured. "
                "Coral bleaching devastated 16% of the world's reefs. "
                "Your location's 1998 value is highlighted on the chart below.",
         "chips":[("Hottest year 1998","chip-record"),("Coral mass bleaching","chip-hot")],
         "anomaly":("🌡️ Record Global Temperature — 1998",
                    "Global mean surface temperature anomaly: +0.62°C vs 1951–1980 baseline. Check how your selected location compares in the time-series chart above.")},
        {"year":"2012","step":2012,"title":"Arctic Sea Ice Minimum",
         "body":"September 2012: Arctic sea ice hit its lowest ever — 3.4 million km², ~50% below the 1979–2000 average. "
                "Arctic Amplification drives polar locations to warm 4× faster than the global mean. "
                "If your location is in the high latitudes, the warming trend is especially steep.",
         "chips":[("Record sea ice loss","chip-hot"),("Arctic amplification","chip-record")],
         "anomaly":("🧊 Sea Ice Collapse — 2012",
                    "The Northwest Passage became navigable for the first time. Permafrost thaw began releasing stored carbon. Polar and sub-polar locations show the strongest warming signature in the trend line.")},
        {"year":"2016","step":2016,"title":"Hottest Year on Record (Then)",
         "body":"2016 set a new record: global surface temperatures 1.1°C above pre-industrial. "
                "The Paris Agreement had just been signed. Your location's 2016 value is marked on every chart.",
         "chips":[("Global temp record","chip-record"),("+1.1°C above baseline","chip-hot")],
         "anomaly":("🔥 Paris Agreement Context — 2016",
                    "Monthly temperatures briefly exceeded 1.5°C above pre-industrial on multiple months. The Great Barrier Reef suffered its worst bleaching: 29% of shallow coral died.")},
        {"year":"2023–2024","step":2023,"title":"The Heat Emergency",
         "body":"2023 broke temperature records on every inhabited continent. "
                "2024 became the first full calendar year exceeding 1.5°C above pre-industrial — the Paris target. "
                "The trend at your location encapsulates the urgency of this era.",
         "chips":[("First >1.5°C year","chip-record"),("All-time heat records","chip-hot"),("Paris limit breached","chip-hot")],
         "anomaly":("🚨 1.5°C Threshold Crossed — 2024",
                    "Verified by GISTEMP, HadCRUT5, Berkeley Earth, ERA5. Not a projection — a measured fact. Your location's recent values reflect this global shift.")},
    ]

    if "story_idx" not in st.session_state:
        st.session_state.story_idx = 0

    n1,n2,n3,n4,n5 = st.columns([1,1,2,1,1])
    with n1:
        if st.button("⏮ First"): st.session_state.story_idx = 0
    with n2:
        if st.button("← Prev") and st.session_state.story_idx > 0: st.session_state.story_idx -= 1
    with n3:
        sel = st.select_slider("Jump to",options=[s["year"] for s in STORIES],
                               value=STORIES[st.session_state.story_idx]["year"],
                               label_visibility="collapsed")
        for i,s in enumerate(STORIES):
            if s["year"] == sel: st.session_state.story_idx = i; break
    with n4:
        if st.button("Next →") and st.session_state.story_idx < len(STORIES)-1: st.session_state.story_idx += 1
    with n5:
        if st.button("Last ⏭"): st.session_state.story_idx = len(STORIES)-1

    idx   = st.session_state.story_idx
    story = STORIES[idx]
    st.progress((idx+1)/len(STORIES), text=f"Chapter {idx+1} of {len(STORIES)}")
    st.markdown("<br>", unsafe_allow_html=True)

    chips_html   = "".join(f'<span class="chip {c[1]}">{c[0]}</span>' for c in story["chips"])
    anomaly_html = (f'<div class="anomaly-card"><div class="anomaly-title">{story["anomaly"][0]}</div>'
                    f'<div class="anomaly-desc">{story["anomaly"][1]}</div></div>') if story["anomaly"] else ""
    st.markdown(f"""
    <div class="story-card">
      <div class="story-year">{story["year"]}</div>
      <div class="story-heading">{story["title"]}</div>
      <div class="story-body">{story["body"]}</div>
      <div style="margin-top:12px;">{chips_html}</div>
      {anomaly_html}
    </div>""", unsafe_allow_html=True)

    # Global anomaly bar chart — highlight story year + mark selected location's value
    st.markdown(f"##### Global Temperature Anomaly 1950–2024 — ⭐ = {g_loc_name} annual value")
    years_s    = np.arange(1950, 2025)
    rng_s      = np.random.default_rng(99)
    trend_s    = (years_s - 1950) * 0.018
    enso_bump  = np.array([0.28 if y in (1983,1988,1998,2010,2016,2023) else 0.0 for y in years_s])
    anom_vals  = trend_s + enso_bump + rng_s.normal(0, 0.07, len(years_s))
    hl_year    = story["step"]

    bar_colors = []
    for i,y in enumerate(years_s):
        if y == hl_year or (hl_year==2023 and y>=2023): bar_colors.append("rgba(0,212,170,1.0)")
        elif anom_vals[i] > 0.5: bar_colors.append("rgba(239,68,68,0.8)")
        elif anom_vals[i] > 0.2: bar_colors.append("rgba(245,158,11,0.7)")
        else: bar_colors.append("rgba(59,130,246,0.6)")

    fig_story = go.Figure()
    fig_story.add_trace(go.Bar(x=years_s, y=anom_vals, marker_color=bar_colors,
        name="Global anomaly", hovertemplate="%{x}<br><b>%{y:+.3f}°C</b><extra></extra>"))
    fig_story.add_trace(go.Scatter(
        x=years_s, y=np.polyval(np.polyfit(years_s,anom_vals,1),years_s),
        mode="lines", name="Trend", line=dict(color="#ef4444",width=2,dash="dot")))

    # Overlay selected location's annual temperature as a line (normalised for comparison)
    ann_loc = df_ts_full.groupby("year")["value"].mean().reset_index()
    ann_loc_norm = (ann_loc["value"] - ann_loc["value"].mean()) / ann_loc["value"].std() * anom_vals.std()
    fig_story.add_trace(go.Scatter(
        x=ann_loc["year"], y=ann_loc_norm,
        mode="lines", name=f"{g_loc_name} (normalised)",
        line=dict(color="#00d4aa", width=2, dash="dot"),
        hovertemplate=f"%{{x}}<br><b>{g_loc_name}: %{{y:+.3f}} (norm)</b><extra></extra>",
    ))

    hline(fig_story, 1.5, color="rgba(245,158,11,0.6)", dash="dash", width=1,
          label="1.5°C threshold", label_color="#f59e0b")
    vline(fig_story, g_year, color="rgba(0,212,170,0.5)", dash="dot", width=1,
          label=str(g_year), label_color="#00d4aa")
    fig_story.update_layout(**DARK,height=270,showlegend=True,
        legend=dict(font=dict(color="#94a3b8",size=11),bgcolor="rgba(10,18,35,0.7)"),
        yaxis_title="Anomaly (°C vs 1950–1980 baseline)",bargap=0.15)
    st.plotly_chart(fig_story, use_container_width=True)

    # Local story heatmap for this chapter's year
    st.markdown(f"##### Global Heatmap — {meta['label']} in {hl_year} · ⭐ = {g_loc_name}")
    ev_idx_yr = min(hl_year, 2024)
    df_ev = get_spatial_slice(g_var, ev_idx_yr, g_month)
    ev_loc_val = float(generate_synthetic_dataset()[g_var].sel(
        lat=g_lat, lon=g_lon, method="nearest").sel(year=ev_idx_yr, month=g_month).values)

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scattergeo(
        lat=df_ev["lat"],lon=df_ev["lon"],mode="markers",
        marker=dict(size=5,color=df_ev["value"],colorscale=colorscale,showscale=True,opacity=0.82,
                    colorbar=dict(title=dict(text=f"{meta['label']} ({meta['unit']})",font=dict(color="#94a3b8",size=11)),
                                  tickfont=dict(color="#94a3b8",size=10),len=0.7,x=1.01)),
        hovertemplate=f"%{{lat:.1f}}°, %{{lon:.1f}}°<br><b>%{{marker.color:.2f}} {meta['unit']}</b><extra></extra>",
        name=meta["label"],
    ))
    fig_ev.add_trace(go.Scattergeo(lat=[g_lat],lon=[g_lon],mode="markers+text",
        marker=dict(size=16,color="#00d4aa",symbol="star",line=dict(width=2,color="white")),
        text=[f"  {g_loc_name}: {ev_loc_val:.1f}{meta['unit']}"],
        textfont=dict(color="#00d4aa",size=11),textposition="middle right",
        name="Selected", hoverinfo="text"))
    fig_ev.update_geos(showland=True,landcolor="rgba(20,35,55,0.85)",
        showocean=True,oceancolor="rgba(5,15,35,0.80)",
        showcoastlines=True,coastlinecolor="rgba(100,116,139,0.4)",coastlinewidth=0.5,
        showframe=False,projection_type="natural earth",bgcolor="rgba(0,0,0,0)")
    fig_ev.update_layout(paper_bgcolor="rgba(0,0,0,0)",geo_bgcolor="rgba(0,0,0,0)",
        height=420,margin=dict(l=0,r=0,t=40,b=0),
        title=dict(text=f"<b>{meta['label']}</b> — {MONTH_NAMES[g_month-1]} {ev_idx_yr}  ⭐ = {g_loc_name}",
                   font=dict(size=14,color="#e2e8f0"),x=0.01),
        showlegend=False)
    st.plotly_chart(fig_ev, use_container_width=True)



#  TAB 6 — AI ASSISTANT

with tab6:
    st.markdown("#### 🤖 App Assistant")
    st.markdown("Ask questions about climate data, anomalies, or how to use the app!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me about climate data or the app..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        response = generate_response(prompt)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)



#  TAB 7 — CLIMATE PREDICTIONS

with tab7:
    st.markdown("#### 🔮 Climate Trend Predictions")
    st.markdown("Machine learning predictions for future climate trends at your selected location.")
    
    # Get time series data
    df_ts = get_time_series(g_lat, g_lon, g_var, 1950)
    years = df_ts["year"].values.reshape(-1, 1)
    values = df_ts["value"].values
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(years, values)
    
    # Predict future years
    future_years = np.arange(2025, 2051).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    # Calculate confidence intervals (simplified)
    residuals = values - model.predict(years)
    std_error = np.std(residuals, ddof=2)
    confidence_interval = 1.96 * std_error  # 95% confidence
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'year': future_years.flatten(),
        'predicted': predictions,
        'lower_bound': predictions - confidence_interval,
        'upper_bound': predictions + confidence_interval
    })
    
    # Plot observed + predicted
    fig_pred = go.Figure()
    
    # Observed data
    fig_pred.add_trace(go.Scatter(
        x=df_ts["year"], y=df_ts["value"],
        mode='lines+markers', name='Observed Data (1950–2024)',
        line=dict(color='#00d4aa', width=2)
    ))
    
    # Predicted trend
    fig_pred.add_trace(go.Scatter(
        x=pred_df["year"], y=pred_df["predicted"],
        mode='lines', name='Predicted Trend (2025–2050)',
        line=dict(color='#ef4444', width=3, dash='dash')
    ))
    
    # Confidence interval
    fig_pred.add_trace(go.Scatter(
        x=pred_df["year"].tolist() + pred_df["year"].tolist()[::-1],
        y=pred_df["upper_bound"].tolist() + pred_df["lower_bound"].tolist()[::-1],
        fill='toself', fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
    
    fig_pred.update_layout(
        title=f"{meta['label']} Trend Prediction for {g_loc_name}",
        xaxis_title="Year",
        yaxis_title=f"{meta['label']} ({meta['unit']})",
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Prediction metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("2030 Prediction", f"{model.predict([[2030]])[0]:.1f}{meta['unit']}")
    with col2:
        st.metric("2050 Prediction", f"{model.predict([[2050]])[0]:.1f}{meta['unit']}")
    with col3:
        trend_per_decade = model.coef_[0] * 10
        st.metric("Trend/Decade", f"{trend_per_decade:+.2f}{meta['unit']}")
    
    st.markdown("**Note:** Predictions use linear regression on historical data. Confidence intervals show uncertainty range.")



#  TAB 8 — EXTREME EVENTS

with tab8:
    st.markdown("#### ⚠️ Extreme Climate Event Detection")
    st.markdown(f"Automatic detection of extreme events at {g_loc_name} based on statistical thresholds.")
    
    # Get time series data
    df_ts = get_time_series(g_lat, g_lon, g_var, 1950)
    values = df_ts["value"].values
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Detect extreme events
    extreme_events = []
    for i, row in df_ts.iterrows():
        val = row["value"]
        year = row["year"]
        z_score = (val - mean_val) / std_val
        
        if g_var == "temperature":
            if z_score > 2:
                event_type = "🔥 Heatwave"
                severity = "Extreme" if z_score > 3 else "High"
            elif z_score < -2:
                event_type = "❄️ Cold Spell"
                severity = "Extreme" if z_score < -3 else "High"
            else:
                continue
        elif g_var == "precipitation":
            if z_score > 2:
                event_type = "🌧️ Heavy Rainfall"
                severity = "Extreme" if z_score > 3 else "High"
            elif z_score < -2:
                event_type = "🏜️ Drought"
                severity = "Extreme" if z_score < -3 else "High"
            else:
                continue
        else:  # wind_speed
            if z_score > 2:
                event_type = "💨 High Winds"
                severity = "Extreme" if z_score > 3 else "High"
            else:
                continue
        
        extreme_events.append({
            "year": year,
            "value": val,
            "z_score": z_score,
            "event_type": event_type,
            "severity": severity
        })
    
    # Display detected events
    if extreme_events:
        st.markdown("### Detected Extreme Events")
        for event in extreme_events[-10:]:  # Show last 10 events
            st.markdown(f"**{event['event_type']}** — {event['year']} ({event['severity']})  \n"
                       f"Value: {event['value']:.1f}{meta['unit']} (Z-score: {event['z_score']:+.2f}σ)")
            st.markdown("---")
    else:
        st.markdown("No extreme events detected in the selected time period.")
    
    # Event frequency chart
    if extreme_events:
        event_years = [e["year"] for e in extreme_events]
        event_counts = pd.Series(event_years).value_counts().sort_index()
        
        fig_events = go.Figure()
        fig_events.add_trace(go.Bar(
            x=event_counts.index, y=event_counts.values,
            marker_color='#ef4444', name='Extreme Events'
        ))
        fig_events.update_layout(
            title=f"Extreme Event Frequency at {g_loc_name}",
            xaxis_title="Year",
            yaxis_title="Number of Events",
            height=300
        )
        st.plotly_chart(fig_events, use_container_width=True)
    
    # Threshold explanation
    st.markdown("### Detection Logic")
    st.markdown("Events are detected using statistical thresholds:")
    st.markdown("- **High**: >2σ from mean")
    st.markdown("- **Extreme**: >3σ from mean")
    st.markdown("σ = standard deviation from 1950-2024 baseline")



#  TAB 9 — CLIMATE INSIGHTS

with tab9:
    st.markdown("#### 🧠 AI Climate Insights")
    st.markdown("Automated analysis and key findings from the climate data.")
    
    # Calculate global insights
    ds = generate_synthetic_dataset()
    
    # Temperature insights
    temp_data = ds["temperature"]
    global_temp_trend = temp_data.mean(dim=["lat", "lon"]).polyfit("year", 1)["polyfit_coefficients"].sel(degree=1).mean(dim="month").item()
    arctic_temp = temp_data.sel(lat=slice(60, 90)).mean(dim=["lat", "lon"])
    tropical_temp = temp_data.sel(lat=slice(-10, 10)).mean(dim=["lat", "lon"])
    arctic_warming_rate = arctic_temp.polyfit("year", 1)["polyfit_coefficients"].sel(degree=1).mean(dim="month").item()
    global_warming_rate = global_temp_trend
    
    # Precipitation insights
    precip_data = ds["precipitation"]
    tropical_rainfall_change = precip_data.sel(lat=slice(-10, 10)).mean(dim=["lat", "lon"]).polyfit("year", 1)["polyfit_coefficients"].sel(degree=1).mean(dim="month").item() * 74  # 74 years
    
    # Extreme events frequency
    temp_values = temp_data.values.flatten()
    heatwave_freq = np.sum(temp_values > np.mean(temp_values) + 2*np.std(temp_values)) / len(temp_values) * 100
    
    # Display insights
    st.markdown("### Key Climate Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌡️ Temperature Trends**")
        st.markdown(f"• Global warming rate: {global_warming_rate*10:.2f}°C/decade")
        st.markdown(f"• Arctic warming rate: {arctic_warming_rate*10:.2f}°C/decade ({arctic_warming_rate/global_warming_rate:.1f}× global average)")
        st.markdown(f"• Heatwave frequency: {heatwave_freq:.1f}% of data points")
    
    with col2:
        st.markdown("**🌧️ Precipitation Patterns**")
        st.markdown(f"• Tropical rainfall change: {tropical_rainfall_change:.1f} mm/day since 1950")
        st.markdown("• Increased monsoon intensity in tropical regions")
        st.markdown("• Drought frequency rising in subtropical areas")
    
    st.markdown("### Regional Analysis")
    st.markdown(f"**Current Location: {g_loc_name}**")
    
    # Local insights
    df_ts = get_time_series(g_lat, g_lon, g_var, 1950)
    local_trend = df_ts["trend_slope_per_decade"].iloc[0]
    local_extremes = len([v for v in df_ts["value"] if abs((v - df_ts["value"].mean()) / df_ts["value"].std()) > 2])
    
    st.markdown(f"• Local {g_var.replace('_', ' ')} trend: {local_trend:+.2f}{meta['unit']}/decade")
    st.markdown(f"• Extreme events detected: {local_extremes} years with >2σ anomalies")
    st.markdown(f"• Risk level: {'High' if local_trend > 0.5 else 'Moderate' if local_trend > 0 else 'Low'}")
    
    st.markdown("### Future Projections")
    st.markdown("• Temperature expected to rise 1.5-4.5°C by 2100 (IPCC scenarios)")
    st.markdown("• Extreme weather events will become more frequent")
    st.markdown("• Sea level rise: 0.3-1.1m by 2100")



#  TAB 10 — CLIMATE ANIMATION

with tab10:
    st.markdown("#### 🎬 Climate Anomaly Animation")
    st.markdown("Watch global climate change unfold over time.")
    
    # Create animation data
    years = np.arange(1950, 2025, 5)  # Every 5 years for animation
    animation_data = []
    
    for year in years:
        df_anim = get_spatial_slice(g_var, year, 6)  # June for consistency
        df_anim["year"] = year
        animation_data.append(df_anim)
    
    df_animation = pd.concat(animation_data)
    
    # Create animated scattergeo
    fig_anim = go.Figure()
    
    fig_anim.add_trace(go.Scattergeo(
        lat=df_animation["lat"], 
        lon=df_animation["lon"],
        mode="markers",
        marker=dict(
            size=3,
            color=df_animation["value"],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=f"{meta['label']} ({meta['unit']})")
        ),
        hovertemplate=f"%{{lat:.1f}}°, %{{lon:.1f}}°<br><b>%{{marker.color:.2f}} {meta['unit']}</b><extra></extra>",
        name=meta["label"],
    ))
    
    # Add animation
    fig_anim.update_layout(
        title=f"Global {meta['label']} Evolution (1950–2024)",
        geo=dict(
            showland=True, landcolor="rgba(20,35,55,0.85)",
            showocean=True, oceancolor="rgba(5,15,35,0.80)",
            showcoastlines=True, coastlinecolor="rgba(100,116,139,0.4)",
            showframe=False, projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)"
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                         method="animate",
                         args=[None, dict(mode="immediate",
                                         frame=dict(duration=1000, redraw=True),
                                         fromcurrent=True,
                                         transition=dict(duration=300))]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], dict(mode="immediate",
                                          frame=dict(duration=0, redraw=False),
                                          transition=dict(duration=0))])]
        )]
    )
    
    # Create frames
    frames = []
    for year in years:
        frame_data = df_animation[df_animation["year"] == year]
        frame = go.Frame(
            data=[go.Scattergeo(
                lat=frame_data["lat"], 
                lon=frame_data["lon"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=frame_data["value"],
                    colorscale=colorscale,
                    showscale=False
                )
            )],
            name=str(year)
        )
        frames.append(frame)
    
    fig_anim.frames = frames
    
    st.plotly_chart(fig_anim, use_container_width=True)
    
    st.markdown("**Animation Controls:** Use the play/pause buttons to watch climate change over time.")
    st.markdown("**Note:** Animation shows data every 5 years for performance. Anomalies become more extreme over time.")



#  TAB 11 — INTERACTIVE ANALYSIS

with tab11:
    st.markdown("#### 🖱️ Interactive Climate Analysis")
    st.markdown("Select a location to analyze detailed climate data and trends.")
    
    # Quick location selector
    quick_locations = {
        "New York, USA": (40.7, -74.0),
        "London, UK": (51.5, -0.1),
        "Tokyo, Japan": (35.7, 139.7),
        "Sydney, Australia": (-33.9, 151.2),
        "Mumbai, India": (19.1, 72.9),
        "Rio de Janeiro, Brazil": (-22.9, -43.2),
        "Cairo, Egypt": (30.0, 31.2),
        "Moscow, Russia": (55.8, 37.6),
        "Cape Town, South Africa": (-33.9, 18.4),
        "Vancouver, Canada": (49.3, -123.1),
        "Arctic Circle": (66.5, 0.0),
        "Antarctica": (-80.0, 0.0),
        "Amazon Rainforest": (-3.1, -60.0),
        "Sahara Desert": (23.0, 13.0),
        "Himalayas": (28.0, 86.9),
        "Pacific Ocean": (0.0, -160.0)
    }
    
    selected_location = st.selectbox(
        "Select location:", 
        ["Custom Coordinates"] + list(quick_locations.keys()),
        help="Choose a preset location or select 'Custom Coordinates' to enter your own"
    )
    
    if selected_location == "Custom Coordinates":
        col1, col2 = st.columns(2)
        with col1:
            analysis_lat = st.number_input("Latitude", -90.0, 90.0, 25.0, step=0.1)
        with col2:
            analysis_lon = st.number_input("Longitude", -180.0, 180.0, 0.0, step=0.1)
    else:
        analysis_lat, analysis_lon = quick_locations[selected_location]
        st.info(f"📍 Analyzing: {selected_location} ({analysis_lat:.1f}°N, {analysis_lon:.1f}°E)")
    
    # Create map showing selected location
    df_analysis = get_spatial_slice(g_var, g_year, g_month)
    
    fig_analysis = go.Figure()
    fig_analysis.add_trace(go.Scattergeo(
        lat=df_analysis["lat"], lon=df_analysis["lon"], mode="markers",
        marker=dict(size=5, color=df_analysis["value"], colorscale=colorscale,
                   showscale=True, colorbar=dict(title=f"{meta['label']} ({meta['unit']})")),
        hovertemplate=f"Lat: %{{lat:.1f}}°, Lon: %{{lon:.1f}}°<br><b>{meta['label']}: %{{marker.color:.2f}} {meta['unit']}</b><extra></extra>",
        name=meta["label"],
    ))
    
    # Add selected location marker - this should update with analysis_lat/lon
    fig_analysis.add_trace(go.Scattergeo(
        lat=[analysis_lat], lon=[analysis_lon], mode="markers+text",
        marker=dict(size=25, color="#00d4aa", symbol="star", line=dict(width=3, color="white")),
        text=[f"📍 {selected_location}"], textposition="top center",
        textfont=dict(color="#00d4aa", size=12, family="Arial Black"),
        name="Selected Location",
        hovertemplate=f"<b>{selected_location}</b><br>Lat: {analysis_lat:.1f}°, Lon: {analysis_lon:.1f}°<extra></extra>"
    ))
    
    fig_analysis.update_geos(
        showland=True, landcolor="rgba(20,35,55,0.85)",
        showocean=True, oceancolor="rgba(5,15,35,0.80)",
        showcoastlines=True, coastlinecolor="rgba(100,116,139,0.4)",
        showframe=False, projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)"
    )
    fig_analysis.update_layout(
        title=f"Analysis Location — {meta['label']} {MONTH_NAMES[g_month-1]} {g_year}",
        height=400
    )
    
    st.plotly_chart(fig_analysis, use_container_width=True)
    
    # Analysis for selected location
    st.markdown(f"### 📍 Climate Analysis for {selected_location}")
    if selected_location == "Custom Coordinates":
        st.markdown(f"**Coordinates**: {analysis_lat:.1f}°N, {analysis_lon:.1f}°E")
    
    # Get time series for selected location
    df_analysis_ts = get_time_series(analysis_lat, analysis_lon, g_var, 1950)
    
    # Calculate trends and statistics
    analysis_trend = df_analysis_ts["trend_slope_per_decade"].iloc[0]
    analysis_mean = df_analysis_ts["value"].mean()
    analysis_current = df_analysis_ts[df_analysis_ts["year"] == g_year]["value"].iloc[0]
    analysis_anomaly = analysis_current - analysis_mean
    analysis_std = df_analysis_ts["value"].std()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Value", f"{analysis_current:.1f}{meta['unit']}")
    with col2:
        st.metric("Trend/Decade", f"{analysis_trend:+.2f}{meta['unit']}")
    with col3:
        st.metric("Anomaly", f"{analysis_anomaly:+.2f}{meta['unit']}")
    with col4:
        z_score = analysis_anomaly / analysis_std if analysis_std > 0 else 0
        st.metric("Z-Score", f"{z_score:+.1f}σ")
    
    # Time series plot
    fig_analysis_ts = go.Figure()
    fig_analysis_ts.add_trace(go.Scatter(
        x=df_analysis_ts["year"], y=df_analysis_ts["value"],
        mode="lines+markers", name="Historical Data",
        line=dict(color="#00d4aa")
    ))
    fig_analysis_ts.add_trace(go.Scatter(
        x=df_analysis_ts["year"], 
        y=df_analysis_ts["year"] * analysis_trend/10 + (analysis_mean - 1977.5 * analysis_trend/10),
        mode="lines", name="Trend Line",
        line=dict(color="#ef4444", dash="dot")
    ))
    # Add current year marker
    fig_analysis_ts.add_trace(go.Scatter(
        x=[g_year], y=[analysis_current],
        mode="markers", name="Current Year",
        marker=dict(size=10, color="#f59e0b", symbol="diamond")
    ))
    fig_analysis_ts.update_layout(
        title=f"{meta['label']} Time Series at {selected_location}",
        xaxis_title="Year", yaxis_title=f"{meta['label']} ({meta['unit']})",
        height=350
    )
    st.plotly_chart(fig_analysis_ts, use_container_width=True)
    
    # Climate insights
    st.markdown("### Climate Insights")
    
    # Trend analysis
    if analysis_trend > 0:
        trend_desc = "warming" if g_var == "temperature" else "increasing"
        trend_severity = "rapidly" if abs(analysis_trend) > analysis_std else "steadily"
        st.markdown(f"• **Trend**: Location is {trend_severity} {trend_desc} at {abs(analysis_trend):.2f}{meta['unit']}/decade")
    else:
        trend_desc = "cooling" if g_var == "temperature" else "decreasing"
        st.markdown(f"• **Trend**: Location shows {trend_desc} trends ({analysis_trend:+.2f}{meta['unit']}/decade)")
    
    # Anomaly analysis
    if abs(z_score) > 2:
        anomaly_desc = "significantly warmer" if analysis_anomaly > 0 else "significantly cooler" if g_var == "temperature" else "much wetter" if analysis_anomaly > 0 else "much drier"
        st.markdown(f"• **Current Conditions**: {anomaly_desc} than average (Z-score: {z_score:+.1f}σ)")
    elif abs(z_score) > 1:
        anomaly_desc = "warmer" if analysis_anomaly > 0 else "cooler" if g_var == "temperature" else "wetter" if analysis_anomaly > 0 else "drier"
        st.markdown(f"• **Current Conditions**: Moderately {anomaly_desc} than average")
    else:
        st.markdown("• **Current Conditions**: Near average for this location")
    
    # Extreme events check
    extreme_years = [row["year"] for _, row in df_analysis_ts.iterrows() 
                    if abs((row["value"] - analysis_mean) / analysis_std) > 2]
    if extreme_years:
        st.markdown(f"• **Extreme Events**: {len(extreme_years)} years with extreme conditions ({', '.join(map(str, extreme_years[-3:]))}...)")
    
    # Regional context
    st.markdown("### Regional Context")
    st.markdown("Compare with global averages and regional patterns:")
    
    # Global comparison
    global_avg = df_analysis["value"].mean()
    location_vs_global = analysis_current - global_avg
    
    st.markdown(f"• **vs Global Average**: {location_vs_global:+.2f}{meta['unit']} ({'above' if location_vs_global > 0 else 'below'} global mean)")
    
    # Latitude-based insights
    lat_desc = "Northern" if analysis_lat > 0 else "Southern"
    if abs(analysis_lat) > 60:
        st.markdown(f"• **Polar Region** ({lat_desc} Hemisphere): Experiences amplified climate change effects")
    elif abs(analysis_lat) < 30:
        st.markdown(f"• **Tropical Region**: Generally stable temperatures with variable precipitation")
    else:
        st.markdown(f"• **Temperate Region** ({lat_desc} Hemisphere): Moderate climate variability")



#  TAB 12 — CLIMATE SIMULATOR

with tab12:
    st.markdown("#### 🌍 Climate Twin Simulation")
    st.markdown("Adjust CO₂ emission levels to simulate future climate impacts.")
    
    # CO2 slider
    co2_level = st.slider(
        "CO₂ Concentration (ppm)", 
        min_value=280, max_value=1000, value=420, step=10,
        help="Current level: ~420 ppm. Pre-industrial: 280 ppm"
    )
    
    # Calculate impacts based on simple climate sensitivity
    # Using approximate relationships:
    # Temperature increase: ~0.8°C per 100 ppm CO2 doubling (simplified)
    # Sea level rise: ~3.3mm per year globally (accelerating)
    # Ice loss: ~267 Gt per year (accelerating)
    
    co2_increase = co2_level - 280  # Above pre-industrial
    temp_increase = co2_increase * 0.008  # ~0.8°C per doubling (560 ppm increase)
    sea_level_rise = 0.1 + (co2_increase / 100) * 0.05  # Base + acceleration
    ice_loss_rate = 267 + (co2_increase / 100) * 50  # Base + acceleration
    
    st.markdown("### Projected Climate Impacts")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "🌡️ Global Temperature Increase", 
            f"+{temp_increase:.1f}°C",
            help="Compared to pre-industrial levels"
        )
    with col2:
        st.metric(
            "🌊 Sea Level Rise (2100)", 
            f"+{sea_level_rise:.1f}m",
            help="Projected rise by 2100"
        )
    with col3:
        st.metric(
            "🧊 Annual Ice Loss", 
            f"{ice_loss_rate:,.0f} Gt/year",
            help="Greenland + Antarctic ice sheet loss"
        )
    
    # Impact visualization
    st.markdown("### Impact Scenarios")
    
    scenarios = {
        "Low Emissions (280 ppm)": {"temp": 0, "sea": 0.1, "ice": 267},
        "Current (420 ppm)": {"temp": temp_increase, "sea": sea_level_rise, "ice": ice_loss_rate},
        "High Emissions (560 ppm)": {"temp": 1.4, "sea": 0.35, "ice": 367},
        "Extreme (700 ppm)": {"temp": 2.1, "sea": 0.6, "ice": 467}
    }
    
    scenario_names = list(scenarios.keys())
    selected_scenario = st.selectbox("Compare with scenario:", scenario_names, index=1)
    
    comp_temp = scenarios[selected_scenario]["temp"]
    comp_sea = scenarios[selected_scenario]["sea"] 
    comp_ice = scenarios[selected_scenario]["ice"]
    
    st.markdown(f"**Comparison with {selected_scenario}:**")
    diff_temp = temp_increase - comp_temp
    diff_sea = sea_level_rise - comp_sea
    diff_ice = ice_loss_rate - comp_ice
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temperature Difference", f"{diff_temp:+.1f}°C", 
                 delta=f"{diff_temp:+.1f}°C" if abs(diff_temp) > 0.1 else "Similar")
    with col2:
        st.metric("Sea Level Difference", f"{diff_sea:+.1f}m",
                 delta=f"{diff_sea:+.1f}m" if abs(diff_sea) > 0.05 else "Similar")
    with col3:
        st.metric("Ice Loss Difference", f"{diff_ice:+.0f} Gt/year",
                 delta=f"{diff_ice:+.0f}" if abs(diff_ice) > 20 else "Similar")
    
    # Risk assessment
    st.markdown("### Risk Assessment")
    if temp_increase < 1.5:
        risk_level = "🟢 Low Risk"
        risk_desc = "Within Paris Agreement targets"
    elif temp_increase < 2.0:
        risk_level = "🟡 Moderate Risk" 
        risk_desc = "Approaching dangerous warming levels"
    elif temp_increase < 3.0:
        risk_level = "🟠 High Risk"
        risk_desc = "Severe impacts expected"
    else:
        risk_level = "🔴 Extreme Risk"
        risk_desc = "Catastrophic climate impacts"
    
    st.markdown(f"**{risk_level}** — {risk_desc}")
    
    # Mitigation suggestions
    st.markdown("### Mitigation Strategies")
    if co2_level > 500:
        st.markdown("• Immediate CO₂ reduction required")
        st.markdown("• Transition to renewable energy")
        st.markdown("• Carbon capture and storage")
    elif co2_level > 450:
        st.markdown("• Accelerate emission reductions")
        st.markdown("• Implement carbon pricing")
        st.markdown("• Protect and restore forests")
    else:
        st.markdown("• Maintain current reduction trajectory")
        st.markdown("• Continue renewable energy adoption")
        st.markdown("• Support international climate agreements")


