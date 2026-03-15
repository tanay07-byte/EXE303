# 🌍 PyClimaExplorer — TECHNEX'26

> An interactive climate data dashboard built with Python · Streamlit · Plotly · Xarray · Pandas · NumPy

---

## Overview

**PyClimaExplorer** is a rapid-prototype interactive visualizer for climate model data (NetCDF files from models like CESM or ERA5 reanalysis). Built for TECHNEX'26 Hackathon — Hack It Out challenge.

---

## Features

| Feature | Description |
|---|---|
| 🗺️ **Spatial Heatmap** | Global map with anomaly highlighting, 5 color palettes, variable/year/month controls |
| 📈 **Time-Series View** | Temporal analysis at any lat/lon with trend overlay and seasonal climatology |
| 🌐 **3D Globe** | Interactive globe projection with temperature anomaly coloring |
| ⚖️ **Compare Mode** | Side-by-side maps for any two years with zonal mean chart |
| 📖 **Guided Story** | 6-chapter climate story mode with anomaly highlights |
| 🤖 **AI Assistant** | Chatbot for answering questions about climate data and app usage |
| 🔮 **Climate Predictions** | ML-based future trend predictions with confidence intervals |
| ⚠️ **Extreme Events** | Automatic detection of heatwaves, cold spells, and heavy rainfall events |
| 🧠 **Climate Insights** | AI-generated key findings and statistical analysis |
| 🎬 **Climate Animation** | Animated visualization of climate change over time |
| 🖱️ **Interactive Analysis** | Select any of 100+ locations for detailed climate metrics and analysis |
| 🌍 **Climate Simulator** | CO₂ emission slider with temperature, sea level, and ice loss projections |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Using Real NetCDF Data

The app currently uses a synthetic dataset that mimics ERA5 structure.  
To load **real data**, replace the `generate_synthetic_dataset()` function in `app.py`:

```python
import xarray as xr

@st.cache_data
def generate_synthetic_dataset():
    # Replace with your real .nc file:
    ds = xr.open_dataset("era5_temperature_2m.nc")
    return ds
```

### Free Sample Datasets

| Source | URL | Notes |
|---|---|---|
| ERA5 via CDS | https://cds.climate.copernicus.eu | Free account required |
| NOAA NCEI | https://www.ncei.noaa.gov/access/search/dataset-search | Public domain |
| NCAR CESM | https://www.cesm.ucar.edu/models/cesm2/datasets/ | Free registration |
| OpenClimateData | https://open-meteo.com/en/docs/climate-api | REST API available |

---

## Project Structure

```
pyclimaexplorer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — Web dashboard framework
- **Plotly** — Interactive charts (scatter geo, bar, line)
- **Xarray** — NetCDF data handling (N-D labeled arrays)
- **Pandas** — Tabular data manipulation
- **NumPy** — Numerical operations

---

## Team

Built for **Hack It Out** @ TECHNEX'26  
IIT (BHU) Varanasi · 14 Mar – 15 Mar 2026

---

## Evaluation Checklist

- [x] Web interface with variable + time range selection
- [x] Spatial View: global heatmap
- [x] Temporal View: time-series with trend
- [x] Documentation (this README)
- [x] **BONUS**: 3D globe visualization
- [x] **BONUS**: Comparison mode (1990 vs 2020)
- [x] **BONUS**: Guided Story Mode with climate anomalies
