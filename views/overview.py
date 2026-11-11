import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import *

df = load_daily()
y0, y1 = sidebar_years(df)
d = df[df.year.between(y0, y1)]
st.title("Sri Lanka climate, at a glance")
st.caption(f"{d.district.nunique()} districts · {y0}–{y1} · daily observations")

# annual national aggregates (district means; rainfall = annual total)
ann = d.groupby(["district", "year"]).agg(temp_mean=("temp_mean", "mean"), temp_min=("temp_min", "mean"),
      temp_max=("temp_max", "mean"), rainfall_mm=("rainfall_mm", "sum"), humidity_mean=("humidity_mean", "mean"),
      solar_radiation_mean=("solar_radiation_mean", "mean")).reset_index()
nat = ann.groupby("year")[BASE].mean().reset_index()

metric_accent([COLORS[c] for c in ["temp_mean", "temp_max", "rainfall_mm", "humidity_mean", "solar_radiation_mean"]])
c = st.columns(5)
c[0].metric("Mean temperature", f"{d.temp_mean.mean():.1f} °C")
c[1].metric("Hottest year", f"{int(nat.loc[nat.temp_max.idxmax(), 'year'])}", f"{nat.temp_max.max():.1f} °C avg max", delta_color="off")
c[2].metric("Annual rainfall", f"{nat.rainfall_mm.mean():,.0f} mm", f"wettest {int(nat.loc[nat.rainfall_mm.idxmax(), 'year'])}", delta_color="off")
c[3].metric("Humidity", f"{d.humidity_mean.mean():.0f} %")
c[4].metric("Solar radiation", f"{d.solar_radiation_mean.mean():.0f} W/m²")

st.subheader("Where is it hottest, wettest, most humid?")
opts = {"Mean temperature": ("temp_mean", "mean", "Oranges"), "Annual rainfall": ("rainfall_mm", "sum", "Blues"),
        "Humidity": ("humidity_mean", "mean", "Teal"), "Solar radiation": ("solar_radiation_mean", "mean", "YlOrBr")}
pick = st.radio("Map variable", list(opts), horizontal=True, label_visibility="collapsed")
col, how, scale = opts[pick]
m = ann.groupby("district")[col].mean().reset_index()  # rainfall already annual total
L, R = st.columns([1.1, 1])
L.plotly_chart(district_map(m, col, f"{LABELS[col]} · {y0}–{y1} average", scale), width="stretch")
r = m.sort_values(col)
fig = go.Figure(go.Bar(x=r[col], y=r.district, orientation="h", marker_color=COLORS[col]))
fig.update_layout(title="District ranking", height=430)
with R:
    show(fig, h=430, legend=False)

st.subheader("How the last decade moved")
t1, t2 = st.columns(2)
with t1:
    show(line(nat, "year", ["temp_max", "temp_mean", "temp_min"], title="Temperature, national annual mean"))
with t2:
    f = go.Figure(go.Bar(x=nat.year, y=nat.rainfall_mm, marker_color=COLORS["rainfall_mm"]))
    f.add_hline(y=nat.rainfall_mm.mean(), line_dash="dot", line_color="#888")
    f.update_layout(title="Annual rainfall (district average, mm)")
    show(f, legend=False)
t3, t4 = st.columns(2)
with t3:
    show(line(nat, "year", ["humidity_mean"], title="Humidity (%)"), legend=False)
with t4:
    show(line(nat, "year", ["solar_radiation_mean"], title="Solar radiation (W/m²)"), legend=False)

st.subheader("Seasonal rhythm")
mo = d.groupby(["district", "year", "month"]).agg(t=("temp_mean", "mean"), r=("rainfall_mm", "sum")).reset_index()
mo = mo.groupby("month")[["t", "r"]].mean().reset_index()
f = go.Figure()
f.add_bar(x=MONTHS, y=mo.r, name="Rainfall (mm)", marker_color=COLORS["rainfall_mm"], opacity=.85)
f.add_scatter(x=MONTHS, y=mo.t, name="Mean temperature (°C)", yaxis="y2", line=dict(color=COLORS["temp_mean"], width=3))
f.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="°C"), yaxis_title="mm")
show(f)

st.subheader("Temperature anomaly by month and year")
an = d.groupby(["year", "month"]).temp_anom.mean().reset_index().pivot(index="year", columns="month", values="temp_anom")
an.columns = [MONTHS[i - 1] for i in an.columns]
show(px.imshow(an, color_continuous_scale=[[0, COLORS["temp_min"]], [.5, "#1A1F27"], [1, COLORS["temp_max"]]],
               zmin=-1.5, zmax=1.5, aspect="auto", labels=dict(color="°C vs normal")), h=360)

st.subheader("Key findings")
trend = np.polyfit(nat.year, nat.temp_mean, 1)[0] * 10
dry = ann.groupby("district").rainfall_mm.mean().idxmin(); wet = ann.groupby("district").rainfall_mm.mean().idxmax()
hot = ann.groupby("district").temp_mean.mean().idxmax()
st.markdown(f"""
- National mean temperature is changing at **{trend:+.2f} °C per decade** over {y0}–{y1}.
- **{wet}** is the wettest district ({ann[ann.district == wet].rainfall_mm.mean():,.0f} mm/yr); **{dry}** is the driest ({ann[ann.district == dry].rainfall_mm.mean():,.0f} mm/yr).
- **{hot}** has the highest mean temperature ({ann[ann.district == hot].temp_mean.mean():.1f} °C).
- Wettest year nationally: **{int(nat.loc[nat.rainfall_mm.idxmax(), 'year'])}**; driest: **{int(nat.loc[nat.rainfall_mm.idxmin(), 'year'])}**.
""")
