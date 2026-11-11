import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import *

df = load_daily()
st.title("Rainfall and flood risk")
y0, y1 = sidebar_years(df)
d = df[df.year.between(y0, y1)]
n_years = y1 - y0 + 1

metric_accent([COLORS["rainfall_mm"], COLORS["humidity_mean"], COLORS["api"], COLORS["flood_score"]])
c = st.columns(4)
c[0].metric("Max daily rainfall", f"{d.rainfall_mm.max():.0f} mm", d.loc[d.rainfall_mm.idxmax(), "district"], delta_color="off")
c[1].metric("Rainy days / year", f"{d.wet_day.sum() / d.district.nunique() / n_years:.0f}")
c[2].metric("Extreme-rain days / year", f"{d.extreme_rain.sum() / d.district.nunique() / n_years:.1f}")
c[3].metric("High+Severe flood-risk days", f"{d.flood_risk.isin(['High', 'Severe']).mean() * 100:.1f} %")

tab1, tab2, tab3 = st.tabs(["Rainfall regime", "Extremes", "Flood-risk indicator"])

with tab1:
    L, R = st.columns([1, 1.2])
    tot = d.groupby(["district", "year"]).rainfall_mm.sum().groupby("district").mean().reset_index()
    L.plotly_chart(district_map(tot, "rainfall_mm", "Average annual rainfall (mm)", "Blues", fmt=",.0f"), width="stretch")
    ms = d.groupby(["district", "year", "month"]).rainfall_mm.sum().groupby(["district", "month"]).mean().unstack()
    ms.columns = MONTHS
    ms = ms.loc[tot.sort_values("rainfall_mm", ascending=False).district]
    with R:
        show(px.imshow(ms, aspect="auto", color_continuous_scale=["#11151B", COLORS["rainfall_mm"]],
                       labels=dict(color="mm / month")), h=430)
    ss = d.groupby(["season", "zone", "year", "district"]).rainfall_mm.sum().groupby(["season", "zone"]).mean().reset_index()
    f = px.bar(ss, x="season", y="rainfall_mm", color="zone", barmode="group", category_orders={"season": SEASONS},
               color_discrete_map={"Dry": "#C65345", "Intermediate": "#E6B84C", "Wet": "#397DA5"})
    f.update_layout(title="Seasonal rainfall by climate zone (mm per season)", yaxis_title="mm"); show(f, h=340)

with tab2:
    dist = st.selectbox("District", ["All districts"] + sorted(d.district.unique()), key="exd")
    s = d if dist == "All districts" else d[d.district == dist]
    L, R = st.columns(2)
    yr = s.groupby("year").extreme_rain.sum() / (1 if dist != "All districts" else s.district.nunique())
    f = go.Figure(go.Bar(x=yr.index, y=yr.values, marker_color=COLORS["rainfall_mm"]))
    f.update_layout(title="Extreme-rain days per year (≥ district's 95th percentile of wet days)")
    with L: show(f, h=320, legend=False)
    wd = s[s.wet_day == 1]
    with R:
        show(px.histogram(wd, x="rainfall_mm", nbins=60, log_y=True, color_discrete_sequence=[COLORS["rainfall_mm"]],
                          title="Daily rainfall distribution on wet days (log count)"), h=320, legend=False)
    st.markdown("**Ten heaviest rain days**")
    top = s.nlargest(10, "rainfall_mm")[["date", "district", "rainfall_mm", "rain_3d", "api", "flood_risk"]]
    top["date"] = top.date.dt.date
    st.dataframe(top.rename(columns={"rainfall_mm": "rain (mm)", "rain_3d": "3-day (mm)", "api": "API"}).round(1),
                 hide_index=True, width="stretch")

with tab3:
    st.caption("A rainfall-based proxy, not a hydrological model: the score blends how wet the last 3 days are with the "
               "antecedent precipitation index (API, k = 0.9), each ranked within the district's own history. "
               "Low < 80th pct · Moderate · High > 93rd · Severe > 98.5th.")
    dist = st.selectbox("District", sorted(d.district.unique()), index=sorted(d.district.unique()).index("Ratnapura"), key="fd")
    s = d[d.district == dist]
    f = go.Figure()
    f.add_bar(x=s.date, y=s.rainfall_mm, name="Rainfall (mm)", marker_color=COLORS["rainfall_mm"])
    f.add_scatter(x=s.date, y=s.api, name="API", yaxis="y2", line=dict(color=COLORS["api"], width=1.5))
    hi = s[s.flood_risk.isin(["High", "Severe"])]
    f.add_scatter(x=hi.date, y=[0] * len(hi), mode="markers", name="High / Severe risk",
                  marker=dict(color=FLOOD_COLORS["Severe"], size=6, symbol="triangle-up"))
    f.update_layout(title=f"{dist}: rainfall, API and flood-risk days", yaxis2=dict(overlaying="y", side="right", showgrid=False, title="API"),
                    yaxis_title="mm"); show(f, h=380)
    L, R = st.columns(2)
    risk = d[d.flood_risk.isin(["High", "Severe"])].groupby("district").size().div(d.groupby("district").size()).mul(100).reset_index(name="pct")
    L.plotly_chart(district_map(risk, "pct", "% of days at High / Severe risk", "Blues", fmt=".1f"), width="stretch")
    cal = s.groupby(["year", "month"]).flood_score.max().reset_index().pivot(index="year", columns="month", values="flood_score")
    cal.columns = [MONTHS[i - 1] for i in cal.columns]
    with R:
        show(px.imshow(cal, aspect="auto", color_continuous_scale=["#11151B", COLORS["flood_score"]], zmin=.5, zmax=1,
                       labels=dict(color="peak score"), title=f"{dist}: peak monthly flood-risk score"), h=430)
