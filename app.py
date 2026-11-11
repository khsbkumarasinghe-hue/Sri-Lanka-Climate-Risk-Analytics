import streamlit as st
from utils import css

st.set_page_config(page_title="Sri Lanka Climate Dashboard", page_icon="🌦️", layout="wide")
css()

pages = [
    st.Page("views/overview.py", title="National overview", icon="🇱🇰", default=True),
    st.Page("views/explorer.py", title="District explorer", icon="📍"),
    st.Page("views/drought.py", title="Drought analysis", icon="🏜️"),
    st.Page("views/precip.py", title="Rainfall & flood risk", icon="🌧️"),
    st.Page("views/forecast.py", title="7-day outlook", icon="🔮"),
]
st.navigation(pages).run()
st.sidebar.caption("Data: daily district weather, 2015 – Nov 2026. Indices computed in prepare_data.py.")
