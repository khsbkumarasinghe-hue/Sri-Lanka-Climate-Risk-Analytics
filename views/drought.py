import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import *

df, mo = load_daily(), load_monthly()
st.title("Drought analysis")
st.caption("SPEI = standardised climatic water balance (rainfall − potential evapotranspiration). "
           "PET is estimated from temperature and solar radiation (Makkink). Values are standardised per calendar month "
           "against 2015–2026, so they describe anomalies relative to this period.")

scale = st.sidebar.radio("Accumulation window", [1, 3, 6], index=1, format_func=lambda k: f"{k}-month")
thr = st.sidebar.select_slider("Drought threshold (SPEI below)", [-0.5, -1.0, -1.5], value=-1.0)
idx = f"spei{scale}"
mo = mo.dropna(subset=[idx]).copy()
mo["drought"] = (mo[idx] < thr).astype(int)

latest = mo[mo.ym == mo.ym.max()]
metric_accent([COLORS["spei"], COLORS["dry_spell"], COLORS["rainfall_mm"], COLORS["pet"]])
c = st.columns(4)
c[0].metric("Latest month", f"{latest.ym.iloc[0]:%b %Y}")
c[1].metric("Districts in drought now", f"{int(latest.drought.sum())} / {len(latest)}")
c[2].metric("Longest dry spell", f"{int(df.dry_spell.max())} days", df.loc[df.dry_spell.idxmax(), "district"], delta_color="off")
c[3].metric("Most drought-prone", mo.groupby("district").drought.mean().idxmax())

tab1, tab2, tab3, tab4 = st.tabs(["Where & when", "Seasonal consistency", "District timeline", "Dry spells"])

with tab1:
    ms = sorted(mo.ym.unique())
    pick = st.select_slider("Month", ms, value=ms[-1], format_func=lambda t: f"{t:%b %Y}")
    L, R = st.columns([1, 1.2])
    cur = mo[mo.ym == pick]
    L.plotly_chart(district_map(cur, idx, f"{idx.upper()} · {pick:%b %Y}", SPEI_SCALE, mid=0, fmt=".2f"), width="stretch")
    heat = mo.pivot(index="district", columns="ym", values=idx)
    f = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, zmin=-2.5, zmax=2.5,
                             colorscale=SPEI_SCALE, colorbar=dict(title=idx.upper(), thickness=10)))
    f.update_layout(title="Every district, every month")
    with R:
        show(f, h=430, legend=False)
    st.markdown("**Drought events per year** (district-months below threshold)")
    ev = mo.groupby(["year", "zone"]).drought.sum().reset_index()
    show(px.bar(ev, x="year", y="drought", color="zone", barmode="group",
                color_discrete_map={"Dry": "#C65345", "Intermediate": "#E6B84C", "Wet": "#397DA5"},
                labels={"drought": "district-months"}), h=300)

with tab2:
    st.markdown(f"Share of years in which each calendar month fell below the threshold ({idx.upper()} < {thr}). "
                "High, repeated values mean the drought is **seasonal and consistent**, not a one-off.")
    fr = mo.groupby(["district", "month"]).drought.mean().unstack() * 100
    order = mo.groupby("district").drought.mean().sort_values(ascending=False).index
    fr = fr.loc[order]; fr.columns = MONTHS
    show(px.imshow(fr, aspect="auto", color_continuous_scale=["#11151B", COLORS["spei"]],
                   labels=dict(color="% of years")), h=620)
    zf = mo.groupby(["zone", "month"]).drought.mean().mul(100).reset_index()
    zf["Month"] = zf.month.map(lambda i: MONTHS[i - 1])
    f = px.line(zf, x="Month", y="drought", color="zone", markers=True, category_orders={"Month": MONTHS},
                color_discrete_map={"Dry": "#C65345", "Intermediate": "#E6B84C", "Wet": "#397DA5"})
    f.update_layout(title="Drought frequency by climate zone", yaxis_title="% of years"); show(f, h=330)

with tab3:
    dist = st.selectbox("District", sorted(mo.district.unique()), index=sorted(mo.district.unique()).index("Hambantota"))
    s = mo[mo.district == dist]
    f = go.Figure()
    f.add_bar(x=s.ym, y=s[idx], marker_color=np.where(s[idx] < thr, "#C65345", np.where(s[idx] > 1, COLORS["rainfall_mm"], "#4B5563")))
    f.add_hline(y=thr, line_dash="dot", line_color=COLORS["spei"])
    f.update_layout(title=f"{dist}: {idx.upper()} (red = drought, blue = very wet)", yaxis_title=idx.upper())
    show(f, legend=False)
    L, R = st.columns(2)
    wb = s.melt("ym", ["rain", "pet"], "var", "mm")
    f = px.line(wb, x="ym", y="mm", color="var", color_discrete_map={"rain": COLORS["rainfall_mm"], "pet": COLORS["pet"]})
    f.update_layout(title="Monthly rainfall vs potential evapotranspiration (mm)")
    with L: show(f, h=320)
    with R:
        show(px.scatter(s, x="temp", y=idx, color_discrete_sequence=[COLORS["spei"]],
                        labels={"temp": "Monthly mean temperature (°C)"}), h=320, legend=False)

with tab4:
    ds = df.groupby(["district", "year"]).dry_spell.max().groupby("district").mean().sort_values().reset_index()
    f = go.Figure(go.Bar(x=ds.dry_spell, y=ds.district, orientation="h", marker_color=COLORS["dry_spell"]))
    f.update_layout(title="Average yearly longest dry spell (days with < 1 mm rain)")
    L, R = st.columns(2)
    with L: show(f, h=560, legend=False)
    ms_ = df.groupby(["district", "month"]).dry_spell.max().unstack(); ms_.columns = MONTHS
    with R:
        show(px.imshow(ms_.loc[ds.district[::-1]], aspect="auto", color_continuous_scale=["#11151B", COLORS["dry_spell"]],
                       labels=dict(color="days")), h=560)
