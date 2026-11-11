"""Shared constants, data loading, theme and chart helpers."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- one colour per variable (your palette + unique colours for derived indices) ---
COLORS = {
    "temp_mean": "#E58A3A", "temp_min": "#4A90A4", "temp_max": "#C65345",
    "rainfall_mm": "#397DA5", "humidity_mean": "#69AFA0", "solar_radiation_mean": "#E6B84C",
    # derived
    "humidex": "#B56BAE", "heat_index": "#9C6B4E", "api": "#7C83DB", "pet": "#A3B565",
    "spei": "#D96FA0", "dry_spell": "#9AA0A6", "flood_score": "#5B9BD5", "temp_anom": "#E58A3A",
}
LABELS = {
    "temp_mean": "Mean temperature (°C)", "temp_min": "Minimum temperature (°C)",
    "temp_max": "Maximum temperature (°C)", "rainfall_mm": "Rainfall (mm)",
    "humidity_mean": "Humidity (%)", "solar_radiation_mean": "Solar radiation (W/m²)",
    "humidex": "Humidex", "heat_index": "Heat index (°C)", "api": "Antecedent precipitation index",
    "pet": "Potential evapotranspiration (mm/day)", "dry_spell": "Dry-spell length (days)",
}
BASE = ["temp_mean", "temp_min", "temp_max", "rainfall_mm", "humidity_mean", "solar_radiation_mean"]
FLOOD_COLORS = {"Low": "#2A3340", "Moderate": "#E6B84C", "High": "#E58A3A", "Severe": "#C65345"}
SPEI_SCALE = [[0, "#C65345"], [0.35, "#E6B84C"], [0.5, "#1A1F27"], [0.65, "#69AFA0"], [1, "#397DA5"]]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = ["Northeast Monsoon", "1st Inter-monsoon", "Southwest Monsoon", "2nd Inter-monsoon"]
COORDS = {  # for maps
    "Ampara": (7.30, 81.67), "Anuradhapura": (8.31, 80.40), "Badulla": (6.99, 81.06),
    "Batticaloa": (7.71, 81.69), "Colombo": (6.93, 79.86), "Galle": (6.05, 80.22),
    "Gampaha": (7.09, 79.99), "Hambantota": (6.12, 81.12), "Jaffna": (9.66, 80.02),
    "Kalutara": (6.58, 79.96), "Kandy": (7.29, 80.63), "Kegalle": (7.25, 80.35),
    "Kilinochchi": (9.39, 80.40), "Kurunegala": (7.49, 80.37), "Mannar": (8.98, 79.91),
    "Matale": (7.47, 80.62), "Matara": (5.95, 80.55), "Monaragala": (6.87, 81.35),
    "Mullaitivu": (9.27, 80.81), "Nuwara Eliya": (6.97, 80.77), "Polonnaruwa": (7.94, 81.00),
    "Puttalam": (8.04, 79.84), "Ratnapura": (6.68, 80.40), "Trincomalee": (8.57, 81.23),
    "Vavuniya": (8.75, 80.50)}


@st.cache_data
def load_daily():
    return pd.read_parquet("data/daily.parquet")


@st.cache_data
def load_monthly():
    return pd.read_parquet("data/monthly.parquet")


def complete_years(df):
    n = df.groupby("year").date.nunique()
    return sorted(n[n >= 360].index)


def css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&display=swap');
    h1, h2, h3 {font-family:'Fraunces', Georgia, serif !important; font-weight:500 !important; letter-spacing:-.01em;}
    [data-testid="stMetric"] {background:#11151B; border-left:3px solid var(--accent,#69AFA0); padding:.7rem 1rem; border-radius:4px;}
    [data-testid="stMetricLabel"] p {font-size:.8rem; opacity:.75;}
    .block-container {padding-top:2rem; max-width:1250px;}
    </style>""", unsafe_allow_html=True)


def metric_accent(cols_colors):
    """Colour the left border of metric cards in order, e.g. metric_accent([c1, c2])."""
    rules = "".join(f'div[data-testid="stColumn"]:nth-child({i}) [data-testid="stMetric"]{{--accent:{c}}}'
                    for i, c in enumerate(cols_colors, 1))
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


def style(fig, h=380, legend=True):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=40, b=10), showlegend=legend,
                      legend=dict(orientation="h", y=1.1, x=0), font=dict(family="sans-serif"))
    fig.update_xaxes(gridcolor="#1E242C", zeroline=False)
    fig.update_yaxes(gridcolor="#1E242C", zeroline=False)
    return fig


def show(fig, **kw):
    st.plotly_chart(style(fig, **kw), width="stretch")


def line(df, x, ys, h=380, title=None, names=None):
    fig = go.Figure()
    for y in ys:
        fig.add_scatter(x=df[x], y=df[y], name=(names or LABELS).get(y, y), mode="lines",
                        line=dict(color=COLORS.get(y, "#ccc"), width=1.6))
    fig.update_layout(title=title)
    return fig


def district_map(d, col, title, scale, h=430, fmt=".1f", mid=None):
    """Bubble map of districts. d needs columns district + col."""
    d = d.copy()
    d["lat"] = d.district.map(lambda x: COORDS[x][0])
    d["lon"] = d.district.map(lambda x: COORDS[x][1])
    d["size"] = 14
    fig = px.scatter_map(d, lat="lat", lon="lon", color=col, size="size", size_max=15, hover_name="district",
                         hover_data={"lat": False, "lon": False, "size": False, col: ":" + fmt},
                         color_continuous_scale=scale, color_continuous_midpoint=mid, zoom=6.3,
                         center=dict(lat=7.85, lon=80.7), map_style="carto-darkmatter")
    fig.update_layout(title=title, height=h, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_colorbar=dict(title="", thickness=10))
    return fig


def sidebar_years(df):
    yrs = complete_years(df)
    return st.sidebar.select_slider("Years", yrs, value=(yrs[0], yrs[-1]))
