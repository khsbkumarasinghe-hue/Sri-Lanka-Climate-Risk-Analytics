import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from forecast_model import run_forecast
from prepare_data import dewpoint, humidex, heat_index, makkink_pet
from utils import *

df = load_daily()
st.title("7-day outlook")
st.caption("Machine-learning outlook trained on this dataset's history (gradient boosting, one model per variable and lead time). "
           "It is a statistical projection, not an official forecast. Ranges show the 10th–90th percentile of past errors.")
dist = st.sidebar.selectbox("District", sorted(df.district.unique()), index=4, key="fcd")
d = df[df.district == dist].sort_values("date")


@st.cache_data(show_spinner=False)
def forecast(district):
    return run_forecast(load_daily().query("district == @district"))


with st.spinner(f"Training models for {dist} (first run takes ~20 s, then cached)…"):
    f, met = forecast(dist)

# derived indices for forecast days
f["dewpoint"] = dewpoint(f.temp_mean, f.humidity_mean)
f["humidex"] = humidex(f.temp_mean, f.dewpoint)
f["heat_index"] = heat_index(f.temp_mean.values, f.humidity_mean.values)
f["pet"] = makkink_pet(f.temp_mean, f.solar_radiation_mean)
api, rains, apis, r3 = d.api.iloc[-1], list(d.rainfall_mm.iloc[-2:]), [], []
for r in f.rainfall_mm:  # API recursion: api_t = 0.9 * api_{t-1} + rain_t
    api = 0.9 * api + r; rains.append(r); apis.append(api); r3.append(sum(rains[-3:]))
f["api"], f["rain_3d"] = apis, r3
pct = lambda hist, x: np.searchsorted(np.sort(hist), x) / len(hist)
f["flood_score"] = 0.5 * pct(d.api, f.api) + 0.5 * pct(d.rain_3d, f.rain_3d)
f["flood_risk"] = pd.cut(f.flood_score, [-1, .80, .93, .985, 2], labels=["Low", "Moderate", "High", "Severe"]).astype(str)


def hx_band(x):
    return ("Comfortable", "#69AFA0") if x < 30 else ("Some discomfort", "#E6B84C") if x < 35 else \
           ("Great discomfort", "#E58A3A") if x < 40 else ("Intense discomfort", "#C65345") if x < 46 else ("Dangerous", "#8B1E2D")


def icon(r):
    return "⛈️" if r.rainfall_mm > 25 else "🌧️" if r.rain_prob > .65 else "🌦️" if r.rain_prob > .45 else "🌤️" if r.rain_prob > .25 else "☀️"


# --- day strip ---------------------------------------------------------------
st.session_state.setdefault("fday", 0)
cols = st.columns(7)
for i, r in f.iterrows():
    label = f"{r.date:%a %d %b}\n\n{icon(r)}\n\n{r.temp_max:.0f}° / {r.temp_min:.0f}°\n\n💧 {r.rain_prob * 100:.0f}%"
    if cols[i].button(label, key=f"day{i}", width="stretch",
                      type="primary" if st.session_state.fday == i else "secondary"):
        st.session_state.fday = i
        st.rerun()
r = f.iloc[st.session_state.fday]
st.subheader(f"{dist} · {r.date:%A, %d %B %Y}")

# --- selected-day values -------------------------------------------------------
metric_accent([COLORS[v] for v in ["temp_mean", "temp_min", "temp_max"]])
c = st.columns(3)
for col, v in zip(c, ["temp_mean", "temp_min", "temp_max"]):
    col.metric(LABELS[v], f"{r[v]:.1f}", f"{r[v + '_lo']:.1f} – {r[v + '_hi']:.1f}", delta_color="off")
metric_accent([COLORS[v] for v in ["rainfall_mm", "humidity_mean", "solar_radiation_mean"]])
c = st.columns(3)
c[0].metric("Rain chance (≥ 1 mm)", f"{r.rain_prob * 100:.0f} %", f"expected {r.rainfall_mm:.1f} mm (up to {r.rainfall_mm_hi:.1f})", delta_color="off")
c[1].metric(LABELS["humidity_mean"], f"{r.humidity_mean:.0f}", f"{r.humidity_mean_lo:.0f} – {r.humidity_mean_hi:.0f}", delta_color="off")
c[2].metric(LABELS["solar_radiation_mean"], f"{r.solar_radiation_mean:.0f}", f"{r.solar_radiation_mean_lo:.0f} – {r.solar_radiation_mean_hi:.0f}", delta_color="off")

st.markdown("##### Derived indices")
L, M, R = st.columns([1.1, 1, 1])
band, bc = hx_band(r.humidex)
g = go.Figure(go.Indicator(mode="gauge+number", value=r.humidex, title={"text": f"Humidex · {band}"},
              gauge=dict(axis=dict(range=[10, 50]), bar=dict(color=COLORS["humidex"]),
                         steps=[dict(range=[10, 30], color="#1B2A27"), dict(range=[30, 35], color="#3A3320"),
                                dict(range=[35, 40], color="#3D2A1B"), dict(range=[40, 46], color="#3D1F1C"),
                                dict(range=[46, 50], color="#4A1018")])))
with L:
    show(g, h=250, legend=False)
with M:
    metric_accent([COLORS["heat_index"], COLORS["api"]])
    st.metric(LABELS["heat_index"], f"{r.heat_index:.1f}")
    st.metric(LABELS["api"], f"{r.api:.0f}")
with R:
    metric_accent([COLORS["pet"], FLOOD_COLORS["High"]])
    st.metric(LABELS["pet"], f"{r.pet:.1f}")
    st.metric("Flood-risk indicator", r.flood_risk)

# --- trend chart ---------------------------------------------------------------
st.markdown("##### Recent observations and the week ahead")
v = st.selectbox("Variable", BASE,
                 format_func=LABELS.get)
past = d.tail(30)
fig = go.Figure()
fig.add_scatter(x=past.date, y=past[v], name="Observed", line=dict(color=COLORS[v], width=2))
fig.add_scatter(x=pd.concat([f.date, f.date[::-1]]), y=pd.concat([f[v + "_hi"], f[v + "_lo"][::-1]]), fill="toself",
                fillcolor="rgba(255,255,255,0.10)", line=dict(width=0), name="10–90% range", hoverinfo="skip")
fig.add_scatter(x=pd.concat([past.date.tail(1), f.date]), y=pd.concat([past[v].tail(1), f[v]]), name="Outlook",
                line=dict(color=COLORS[v], width=2, dash="dash"), mode="lines+markers")
fig.add_vline(x=past.date.iloc[-1], line_dash="dot", line_color="#666")
fig.add_scatter(x=[r.date], y=[r[v]], mode="markers", marker=dict(size=13, color="#fff", symbol="circle-open", line=dict(width=2)),
                name="Selected day")
fig.update_layout(title=LABELS[v]); show(fig, h=360)

with st.expander("How reliable is this? Hold-out accuracy (last 365 days)"):
    m = met.groupby("variable")[["mae_model", "mae_climatology"]].mean()
    m["skill vs climatology"] = (1 - m.mae_model / m.mae_climatology).map("{:.0%}".format)
    m.index = [LABELS[i] for i in m.index]
    st.dataframe(m.rename(columns={"mae_model": "MAE (model)", "mae_climatology": "MAE (monthly climatology)"}).round(2),
                 width="stretch")
    st.caption("MAE = mean absolute error averaged over lead times 1–7. Positive skill means the model beats simply "
               "guessing the monthly average. Rainfall is the hardest variable, so use the probability, not the exact mm.")
