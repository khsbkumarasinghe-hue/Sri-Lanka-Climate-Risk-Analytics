import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
from datetime import datetime
import json
import urllib.request

# ==========================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Sri Lanka Climate & Risk Analytics Dashboard",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Unified Color Palette
PALETTE = {
    'temp_max': '#D90429',             # Crimson Red
    'temp_mean': '#E76F51',            # Burnt Orange
    'temp_min': '#F4A261',             # Soft Amber
    'rainfall_mm': '#2A9D8F',          # Deep Teal
    'humidity_mean': '#457B9D',        # Steel Blue
    'solar_radiation_mean': '#E9C46A', # Warm Yellow
    'api_flood': '#1D3557',            # Dark Navy
    'spei_drought': '#C8553D',         # Terracotta Red
    'humidex': '#8E7DBE'              # Muted Violet
}

# CSS Styling Injection
st.markdown(f"""
    <style>
    .metric-card {{
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2A9D8F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .stApp {{
        background-color: #FFFFFF;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA PROCESSING & CACHING ENGINE
# ==========================================

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('sri_lanka_all_districts_weather_2015_onwards.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['datetime'].dt.strftime('%b')
    
    # 1. Hargreaves Potential Evapotranspiration (PET) Approximation
    temp_range = np.maximum(df['temp_max'] - df['temp_min'], 0.1)
    df['pet_mm'] = 0.0023 * (df['solar_radiation_mean'] / 10.0) * (df['temp_mean'] + 17.8) * np.sqrt(temp_range)
    
    # 2. Water Deficit / Surplus
    df['water_balance'] = df['rainfall_mm'] - df['pet_mm']
    
    # 3. Humidex Calculation
    e = 6.11 * (10 ** (7.5 * df['temp_mean'] / (237.3 + df['temp_mean']))) * (df['humidity_mean'] / 100.0)
    df['humidex'] = df['temp_mean'] + (5/9) * (e - 10)
    
    # 4. Calculate Antecedent Precipitation Index (API) per District safely
    df = df.sort_values(['district', 'datetime']).reset_index(drop=True)
    
    def compute_api_series(series, decay=0.85):
        api_vals = []
        current_api = 0.0
        for r in series:
            current_api = r + (decay * current_api)
            api_vals.append(current_api)
        return pd.Series(api_vals, index=series.index)

    df['api_index'] = df.groupby('district')['rainfall_mm'].transform(compute_api_series)
    return df

df_raw = load_and_preprocess_data()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://img.icons8.com/isometric/100/cloud.png", width=70)
st.sidebar.title("Page Navigation")

navigation = st.sidebar.radio(
    "Visit page",
    ["🌐 National Overview", "🗺️ District Analysis", "🏜️ Drought Analysis", "🌧️ Precipitation & Flood", "🔮 Interactive Forecast"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")
min_date = df_raw['datetime'].min().date()
max_date = df_raw['datetime'].max().date()

date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df_raw[(df_raw['datetime'].dt.date >= start_date) & (df_raw['datetime'].dt.date <= end_date)]
else:
    df = df_raw.copy()

# ==========================================
# PAGE 1: NATIONAL OVERVIEW
# ==========================================
if navigation == "🌐 National Overview":
    st.title("🌐 Sri Lanka National Climate Overview")
    st.markdown("Comprehensive 10-year historical climate analytics and aggregated environmental KPIs.")
    
    # Find the row corresponding to the maximum single-day rainfall
    max_rain_row = df.loc[df['rainfall_mm'].idxmax()]
    max_rain_date = max_rain_row['datetime']

    # KPI Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("10-Yr Mean Temperature", f"{df['temp_mean'].mean():.1f} °C")
    with col2:
        st.metric("Max Single-Day Rain", f"{max_rain_row['rainfall_mm']:.1f} mm", delta=f"Date: {max_rain_date}")
    with col3:
        st.metric("Avg Solar Radiation", f"{df['solar_radiation_mean'].mean():.0f} Wh/m²")

    st.markdown("---")
    
    # Visualizations Section
    st.subheader("District Climate Aggregates")
    metric_choice = st.selectbox(
        "Select Aggregation Variable", 
        ['rainfall_mm', 'temp_mean', 'solar_radiation_mean', 'humidity_mean']
    )

    # 1. Dynamically aggregate based on selected variable
    if metric_choice == 'rainfall_mm':
        date_col = 'datetime' if 'datetime' in df.columns else 'Date'
        df_temp = df.copy()
    
        if 'district' not in df_temp.columns:
            df_temp = df_temp.reset_index()
        
        if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])

        df_temp['year_month'] = df_temp[date_col].dt.to_period('M').astype(str)

        monthly_totals = (
            df_temp.groupby(['year_month', 'district'])[metric_choice]
            .sum()
            .reset_index()
        )

        dist_summary = monthly_totals.groupby('district')[metric_choice].mean().reset_index()
        
        y_axis_title = "Mean Monthly Rainfall (mm)"
        chart_title = "Mean Monthly Rainfall by District"
    else:
        dist_summary = df.groupby('district')[metric_choice].mean().reset_index()
        y_axis_title = f"{metric_choice.replace('_', ' ').title()}"
        chart_title = f"{metric_choice.replace('_', ' ').title()} by District"

    dist_summary = dist_summary.sort_values(metric_choice, ascending=False)

    fig_bar = px.bar(
        dist_summary, 
        x='district', 
        y=metric_choice,
        color_discrete_sequence=[PALETTE.get(metric_choice, '#2A9D8F')],
        title=chart_title,
        labels={metric_choice: y_axis_title, 'district': 'District'}
    )

    fig_bar.update_layout(xaxis_tickangle=-45, template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # DISTRICT CLIMATE MAPS
    # ==========================================
    st.markdown("---")
    st.subheader("District Climate Maps")

    with open("geoBoundaries-LKA-ADM2-cleaned.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    date_col = 'datetime' if 'datetime' in df.columns else 'Date'
    df_temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])

    monthly_totals = (
        df_temp.groupby([df_temp[date_col].dt.to_period('M'), 'district'])['rainfall_mm']
        .sum()
        .reset_index()
    )
    rain_summary = monthly_totals.groupby('district')['rainfall_mm'].mean().reset_index()

    temp_summary = df.groupby('district')['temp_mean'].mean().reset_index()

    map_col1, map_col2 = st.columns(2)

    with map_col1:
        fig_rain_map = px.choropleth(
            rain_summary,
            geojson=geojson_data,
            locations='district',
            featureidkey="properties.shapeName",
            locationmode="geojson-id",
            color='rainfall_mm',
            color_continuous_scale='Blues',
            title="Mean Monthly Rainfall by District (mm)",
            labels={'rainfall_mm': 'Rainfall (mm)', 'district': 'District'}
        )
        fig_rain_map.update_geos(fitbounds="locations", visible=False)
        fig_rain_map.update_layout(
            template="plotly_white",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        st.plotly_chart(fig_rain_map, use_container_width=True)

    with map_col2:
        fig_temp_map = px.choropleth(
            temp_summary,
            geojson=geojson_data,
            locations='district',
            featureidkey="properties.shapeName",
            locationmode="geojson-id",
            color='temp_mean',
            color_continuous_scale='OrRd',
            title="Mean Temperature by District (°C)",
            labels={'temp_mean': 'Temp (°C)', 'district': 'District'}
        )
        fig_temp_map.update_geos(fitbounds="locations", visible=False)
        fig_temp_map.update_layout(
            template="plotly_white",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        st.plotly_chart(fig_temp_map, use_container_width=True)

        st.markdown("---")

# PAGE 2: DISTRICT EXPLORER
# ==========================================
elif navigation == "🗺️ District Analysis":
    st.title("🗺️ District Analysis")
    
    selected_district = st.selectbox("Select District", sorted(df_raw['district'].unique()))
    df_dist = df[df['district'] == selected_district]

    df_dist_temp = df_dist.copy()
    date_col = 'datetime' if 'datetime' in df_dist_temp.columns else 'Date'
    if not pd.api.types.is_datetime64_any_dtype(df_dist_temp[date_col]):
        df_dist_temp[date_col] = pd.to_datetime(df_dist_temp[date_col])

    monthly_mean_rain = (
        df_dist_temp.groupby(df_dist_temp[date_col].dt.to_period('M'))['rainfall_mm']
        .sum()
        .mean()
    )
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Max Temperature", f"{df_dist['temp_max'].max():.1f} °C")
    k2.metric("Min Temperature", f"{df_dist['temp_min'].min():.1f} °C")
    k3.metric("Mean Monthly Rain", f"{monthly_mean_rain:.0f} mm")
    k4.metric("Avg Humidity", f"{df_dist['humidity_mean'].mean():.1f} %")
    
    st.markdown("---")
    
    st.subheader(f"Temperature Envelope for {selected_district}")
    
    df_dist_sorted = df_dist.sort_values('datetime')
    fig_ribbon = gg.Figure()
    
    fig_ribbon.add_trace(gg.Scatter(
        x=df_dist_sorted['datetime'], y=df_dist_sorted['temp_max'],
        mode='lines', name='Max Temp', line=dict(color=PALETTE['temp_max'], width=1)
    ))
    fig_ribbon.add_trace(gg.Scatter(
        x=df_dist_sorted['datetime'], y=df_dist_sorted['temp_min'],
        mode='lines', name='Min Temp', fill='tonexty',
        fillcolor='rgba(244, 162, 97, 0.2)', line=dict(color=PALETTE['temp_min'], width=1)
    ))
    fig_ribbon.add_trace(gg.Scatter(
        x=df_dist_sorted['datetime'], y=df_dist_sorted['temp_mean'],
        mode='lines', name='Mean Temp', line=dict(color=PALETTE['temp_mean'], width=2)
    ))
    
    fig_ribbon.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig_ribbon, use_container_width=True)

    st.markdown("---")

    st.subheader(f"Precipitation Trend for {selected_district}")

    df_dist_sorted = df_dist.sort_values('datetime')
    fig_rain_line = gg.Figure()

    fig_rain_line.add_trace(gg.Scatter(
        x=df_dist_sorted['datetime'],
        y=df_dist_sorted['rainfall_mm'],
        mode='lines',
        name='Rainfall',
        fill='tozeroy',
        fillcolor='rgba(42, 157, 143, 0.2)',
        line=dict(color=PALETTE['rainfall_mm'], width=2)
    ))

    fig_rain_line.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        hovermode="x unified"
    )

    st.plotly_chart(fig_rain_line, use_container_width=True)

    st.markdown("---")

    st.subheader(f"Solar Radiation Trend for {selected_district}")

    df_dist_sorted = df_dist.sort_values('datetime')
    fig_solar = gg.Figure()

    fig_solar.add_trace(gg.Scatter(
        x=df_dist_sorted['datetime'],
        y=df_dist_sorted['solar_radiation_mean'],
        mode='lines',
        name='Solar Radiation',
        fill='tozeroy',
        fillcolor='rgba(233, 196, 106, 0.2)',
        line=dict(color=PALETTE['solar_radiation_mean'], width=2)
    ))

    fig_solar.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Solar Radiation (Wh/m²)",
        hovermode="x unified"
    )

    st.plotly_chart(fig_solar, use_container_width=True)

    st.markdown("---")

    st.subheader(f"Relative Humidity Trend for {selected_district}")

    df_dist_sorted = df_dist.sort_values('datetime')
    fig_humidity = gg.Figure()

    fig_humidity.add_trace(gg.Scatter(
        x=df_dist_sorted['datetime'],
        y=df_dist_sorted['humidity_mean'],
        mode='lines',
        name='Mean Humidity',
        line=dict(color=PALETTE['humidity_mean'], width=2)
    ))

    fig_humidity.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Relative Humidity (%)",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified"
    )

    st.plotly_chart(fig_humidity, use_container_width=True)

    st.markdown("---")


# ==========================================
# PAGE 3: DROUGHT ANALYSIS
# ==========================================
elif navigation == "🏜️ Drought Analysis":
    st.title("🏜️ Drought & Water Deficit Analysis")
    st.markdown("Analysis of Potential Evapotranspiration (PET), Water Balance, and Dry Spells.")
    
    selected_district = st.selectbox("Select District for Drought Profiling", sorted(df_raw['district'].unique()), key="drought_dist")
    df_d = df[df['district'] == selected_district].sort_values('datetime')
    
    df_monthly_d = df_d.set_index('datetime').resample('M')[['rainfall_mm', 'pet_mm', 'water_balance']].sum().reset_index()
    
    fig_water = gg.Figure()
    fig_water.add_trace(gg.Bar(
        x=df_monthly_d['datetime'], y=df_monthly_d['rainfall_mm'],
        name='Rainfall (mm)', marker_color=PALETTE['rainfall_mm']
    ))
    fig_water.add_trace(gg.Scatter(
        x=df_monthly_d['datetime'], y=df_monthly_d['pet_mm'],
        name='Evapotranspiration PET (mm)', line=dict(color=PALETTE['spei_drought'], width=2.5)
    ))
    fig_water.update_layout(template="plotly_white", title=f"Monthly Rainfall vs PET Demand ({selected_district})")
    st.plotly_chart(fig_water, use_container_width=True)

# ==========================================
# PAGE 4: PRECIPITATION & FLOOD RISK
# ==========================================
elif navigation == "🌧️ Precipitation & Flood":
    st.title("🌧️ Precipitation & Flood Saturation Analytics")
    
    selected_district = st.selectbox("Select District for Flood Risk Analysis", sorted(df_raw['district'].unique()), key="flood_dist")
    df_f = df[df['district'] == selected_district].sort_values('datetime')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Antecedent Precip Index (API)", f"{df_f['api_index'].iloc[-1]:.1f} mm")
    with col2:
        heavy_rain_days = (df_f['rainfall_mm'] >= 50).sum()
        st.metric("Heavy Rain Days (≥ 50mm)", f"{heavy_rain_days} Days")
        
    fig_api = px.line(
        df_f, x='datetime', y='api_index',
        title=f"Antecedent Precipitation Saturation Index - API ({selected_district})",
        color_discrete_sequence=[PALETTE['api_flood']]
    )
    fig_api.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="High Saturation Threshold")
    fig_api.update_layout(template="plotly_white")
    st.plotly_chart(fig_api, use_container_width=True)

# ==========================================
# PAGE 5: INTERACTIVE FORECAST
# ==========================================
elif navigation == "🔮 Interactive Forecast":
    st.title("🔮 Interactive 7-Day Climate Forecast")
    st.markdown("Pre-computed 7-day recursive XGBoost model predictions across weather variables.")
    
    # 1. Load Pre-calculated Forecasts
    @st.cache_data
    def load_forecast_data():
        try:
            df_fc = pd.read_parquet("forecasts_7days.parquet")
        except Exception:
            df_fc = pd.read_csv("forecasts_7days.csv")
            
        df_fc['datetime'] = pd.to_datetime(df_fc['datetime'])
        
        # Calculate Humidex index for reference
        e = 6.11 * (10 ** (7.5 * df_fc['temp_mean'] / (237.3 + df_fc['temp_mean']))) * (df_fc['humidity_mean'] / 100.0)
        df_fc['humidex'] = df_fc['temp_mean'] + (5/9) * (e - 10)
        return df_fc

    forecast_all = load_forecast_data()
    
    # 2. In-Page District Selector Filter
    target_dist = st.selectbox("Select Target District", sorted(forecast_all['district'].unique()), key="page_fc_dist")
    forecast_df = forecast_all[forecast_all['district'] == target_dist].sort_values('datetime').reset_index(drop=True)
    
    st.markdown("---")

    # 3. Table Output Display
    st.subheader(f"📊 7-Day Forecast Data Table ({target_dist})")
    
    # Prepare styled display table
    display_df = forecast_df.copy()
    display_df['Date'] = display_df['datetime'].dt.strftime('%Y-%m-%d (%a)')
    
    # Rename columns for presentation
    rename_dict = {
        'Date': 'Date',
        'temp_mean': 'Mean Temp (°C)',
        'temp_min': 'Min Temp (°C)',
        'temp_max': 'Max Temp (°C)',
        'rainfall_mm': 'Rainfall (mm)',
        'humidity_mean': 'Humidity (%)',
        'solar_radiation_mean': 'Solar Radiation (Wh/m²)',
        'humidex': 'Humidex (°C)'
    }
    
    table_df = display_df[list(rename_dict.keys())].rename(columns=rename_dict)
    
    # Format float precision for clean table display
    st.dataframe(
        table_df.style.format({
            'Mean Temp (°C)': '{:.1f}',
            'Min Temp (°C)': '{:.1f}',
            'Max Temp (°C)': '{:.1f}',
            'Rainfall (mm)': '{:.1f}',
            'Humidity (%)': '{:.1f}',
            'Solar Radiation (Wh/m²)': '{:.1f}',
            'Humidex (°C)': '{:.1f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # 4. CSV Download
    st.download_button(
        label="📥 Download 7-Day Forecast CSV",
        data=table_df.to_csv(index=False),
        file_name=f"{target_dist}_7day_climate_forecast.csv",
        mime="text/csv"
    )