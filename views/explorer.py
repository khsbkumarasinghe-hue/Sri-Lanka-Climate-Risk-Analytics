import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import *

df = load_daily()
st.title("District explorer")
sel = st.sidebar.selectbox("District", sorted(df.district.unique()), index=4)
y0, y1 = sidebar_years(df)
vars_ = st.sidebar.multiselect("Variables", BASE, default=["temp_mean", "rainfall_mm"], format_func=LABELS.get)
d = df[(df.district == sel) & df.year.between(y0, y1)]
nat = df[df.year.between(y0, y1)]

st.caption(f"{sel} · {d.zone.iloc[0]} Zone")
metric_accent([COLORS[c] for c in ["temp_mean", "temp_max", "rainfall_mm", "humidity_mean", "solar_radiation_mean"]])
ann = d[d.year.isin(complete_years(d))].groupby("year")
c = st.columns(5)
c[0].metric("Mean temp", f"{d.temp_mean.mean():.1f} °C", f"{d.temp_mean.mean() - nat.temp_mean.mean():+.1f} vs national", delta_color="off")
c[1].metric("Max temp record", f"{d.temp_max.max():.1f} °C")
c[2].metric("Annual rainfall", f"{ann.rainfall_mm.sum().mean():,.0f} mm")
c[3].metric("Humidity", f"{d.humidity_mean.mean():.0f} %")
c[4].metric("Solar", f"{d.solar_radiation_mean.mean():.0f} W/m²")

tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Seasonality", "Year × month", "Compare districts"])

with tab1:
    win = st.select_slider("Smoothing", ["Daily", "7-day", "30-day", "90-day"], value="30-day")
    n = {"Daily": 1, "7-day": 7, "30-day": 30, "90-day": 90}[win]
    s = d.set_index("date")[BASE].rolling(n, min_periods=1).mean().reset_index()
    if not vars_:
        st.info("Pick at least one variable in the sidebar.")
    for v in vars_:  # one chart per variable keeps units honest
        show(line(s, "date", [v], h=260, title=LABELS[v]), legend=False, h=260)
    hs = d.set_index("date")[["humidex", "heat_index"]].rolling(n, min_periods=1).mean().reset_index()
    show(line(hs, "date", ["humidex", "heat_index"], title="Heat stress: humidex and heat index"), h=280)
    mm = d.set_index("date").temp_anom.resample("MS").mean().dropna()
    slope = np.polyfit(np.arange(len(mm)), mm.values, 1)[0] * 120
    st.info(f"Warming trend in {sel}: **{slope:+.2f} °C per decade** (linear fit on monthly temperature anomalies).")

with tab2:
    v = st.selectbox("Variable", BASE, format_func=LABELS.get, key="seasv")
    agg = "sum" if v == "rainfall_mm" else "mean"
    mo = d.groupby(["year", "month"])[v].agg(agg).reset_index()
    mo["Month"] = mo.month.map(lambda i: MONTHS[i - 1])
    f = px.box(mo, x="Month", y=v, category_orders={"Month": MONTHS}, color_discrete_sequence=[COLORS[v]])
    f.update_layout(title=f"{LABELS[v]} — monthly {'total' if agg == 'sum' else 'mean'}, spread across years", yaxis_title="")
    show(f, legend=False)
    ss = d.groupby("season")[BASE].mean().reindex(SEASONS)
    ss["rainfall_mm"] = d.groupby(["season", "year"]).rainfall_mm.sum().groupby("season").mean().reindex(SEASONS)
    st.markdown("**Season summary** (rainfall = average seasonal total)")
    st.dataframe(ss.rename(columns=LABELS).round(1), width="stretch")

with tab3:
    v = st.selectbox("Variable", BASE, format_func=LABELS.get, key="hmv")
    agg = "sum" if v == "rainfall_mm" else "mean"
    p = d.groupby(["year", "month"])[v].agg(agg).reset_index().pivot(index="year", columns="month", values=v)
    p.columns = [MONTHS[i - 1] for i in p.columns]
    c0 = COLORS[v]
    show(px.imshow(p, aspect="auto", color_continuous_scale=["#11151B", c0], labels=dict(color=LABELS[v])), h=420)

with tab4:
    others = st.multiselect("Compare with", [x for x in sorted(df.district.unique()) if x != sel],
                            default=["Nuwara Eliya", "Hambantota"])
    v = st.selectbox("Variable", BASE, format_func=LABELS.get, key="cmpv")
    agg = "sum" if v == "rainfall_mm" else "mean"
    sub = df[df.district.isin([sel] + others) & df.year.between(y0, y1)]
    t = sub.groupby(["district", "year", "month"])[v].agg(agg).groupby(["district", "month"]).mean().reset_index()
    t["Month"] = t.month.map(lambda i: MONTHS[i - 1])
    f = px.line(t, x="Month", y=v, color="district", markers=True, category_orders={"Month": MONTHS})
    f.update_layout(title=f"{LABELS[v]} — monthly climatology"); show(f)
