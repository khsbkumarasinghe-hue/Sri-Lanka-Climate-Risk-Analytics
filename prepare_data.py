"""Feature engineering. Run once:  python prepare_data.py [path/to/raw.csv]
Writes data/daily.parquet (daily, per district) and data/monthly.parquet (SPI/SPEI)."""
import sys
import numpy as np
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw.csv"

# --- static metadata ---------------------------------------------------------
COORDS = {  # approx. district centroids (lat, lon)
    "Ampara": (7.30, 81.67), "Anuradhapura": (8.31, 80.40), "Badulla": (6.99, 81.06),
    "Batticaloa": (7.71, 81.69), "Colombo": (6.93, 79.86), "Galle": (6.05, 80.22),
    "Gampaha": (7.09, 79.99), "Hambantota": (6.12, 81.12), "Jaffna": (9.66, 80.02),
    "Kalutara": (6.58, 79.96), "Kandy": (7.29, 80.63), "Kegalle": (7.25, 80.35),
    "Kilinochchi": (9.39, 80.40), "Kurunegala": (7.49, 80.37), "Mannar": (8.98, 79.91),
    "Matale": (7.47, 80.62), "Matara": (5.95, 80.55), "Monaragala": (6.87, 81.35),
    "Mullaitivu": (9.27, 80.81), "Nuwara Eliya": (6.97, 80.77), "Polonnaruwa": (7.94, 81.00),
    "Puttalam": (8.04, 79.84), "Ratnapura": (6.68, 80.40), "Trincomalee": (8.57, 81.23),
    "Vavuniya": (8.75, 80.50)}
WET = ["Colombo", "Gampaha", "Kalutara", "Galle", "Matara", "Ratnapura", "Kegalle", "Nuwara Eliya"]
INTER = ["Kandy", "Matale", "Badulla", "Kurunegala", "Monaragala"]  # approximate classification
ZONE = {d: "Wet" if d in WET else "Intermediate" if d in INTER else "Dry" for d in COORDS}


def season(m):
    return ("Northeast Monsoon" if m in (12, 1, 2) else "1st Inter-monsoon" if m in (3, 4)
            else "Southwest Monsoon" if m in range(5, 10) else "2nd Inter-monsoon")


# --- indices -----------------------------------------------------------------
def dewpoint(t, rh):  # Magnus
    g = np.log(np.clip(rh, 1, 100) / 100) + 17.62 * t / (243.12 + t)
    return 243.12 * g / (17.62 - g)


def humidex(t, td):
    e = 6.11 * np.exp(5417.753 * (1 / 273.16 - 1 / (273.15 + td)))
    return t + 0.5555 * (e - 10)


def heat_index(t, rh):  # Rothfusz regression (NWS); applies above ~27 C
    f = t * 9 / 5 + 32
    hi = (-42.379 + 2.04901523 * f + 10.14333127 * rh - 0.22475541 * f * rh - 6.83783e-3 * f**2
          - 5.481717e-2 * rh**2 + 1.22874e-3 * f**2 * rh + 8.5282e-4 * f * rh**2 - 1.99e-6 * f**2 * rh**2)
    return np.where(t >= 27, (hi - 32) * 5 / 9, t)


def makkink_pet(t, rs_wm2):
    """Makkink reference ET (mm/day) from mean temp and shortwave radiation (W/m2)."""
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    delta = 4098 * es / (t + 237.3) ** 2
    ratio = delta / (delta + 0.066)
    return np.clip(0.61 * ratio * (rs_wm2 * 0.0864) / 2.45 - 0.12, 0, None)


def run_len(is_dry):  # consecutive-days counter
    grp = (~is_dry).cumsum()
    return is_dry.groupby(grp).cumsum()


def zscore_by_month(s, months):
    g = s.groupby(months)
    return (s - g.transform("mean")) / g.transform("std")


def main():
    df = pd.read_csv(RAW, parse_dates=["datetime"]).rename(columns={"datetime": "date"})
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    df["year"], df["month"], df["doy"] = df.date.dt.year, df.date.dt.month, df.date.dt.dayofyear
    df["season"] = df.month.map(season)
    df["zone"] = df.district.map(ZONE)
    df["lat"] = df.district.map(lambda d: COORDS[d][0])
    df["lon"] = df.district.map(lambda d: COORDS[d][1])

    df["dtr"] = df.temp_max - df.temp_min
    df["dewpoint"] = dewpoint(df.temp_mean, df.humidity_mean)
    df["humidex"] = humidex(df.temp_mean, df.dewpoint)
    df["heat_index"] = heat_index(df.temp_mean.values, df.humidity_mean.values)
    df["pet"] = makkink_pet(df.temp_mean, df.solar_radiation_mean)
    df["water_balance"] = df.rainfall_mm - df.pet

    g = df.groupby("district", group_keys=False)
    df["rain_3d"] = g.rainfall_mm.transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["rain_7d"] = g.rainfall_mm.transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["api"] = g.rainfall_mm.transform(lambda s: s.ewm(alpha=0.1, adjust=False).mean() * 10)  # k = 0.9
    df["dry_spell"] = g.rainfall_mm.transform(lambda s: run_len(s < 1.0))
    df["wet_day"] = (df.rainfall_mm >= 1.0).astype(int)
    p95 = df[df.wet_day == 1].groupby("district").rainfall_mm.quantile(0.95)
    df["r95_threshold"] = df.district.map(p95)
    df["extreme_rain"] = (df.rainfall_mm >= df.r95_threshold).astype(int)

    # flood-risk indicator: percentile of API and 3-day rain, within each district
    score = 0.5 * g.api.rank(pct=True) + 0.5 * g.rain_3d.rank(pct=True)
    df["flood_score"] = score
    df["flood_risk"] = pd.cut(score, [0, 0.80, 0.93, 0.985, 1.01],
                              labels=["Low", "Moderate", "High", "Severe"]).astype(str)
    # temperature anomaly vs district monthly climatology
    df["temp_anom"] = df.temp_mean - df.groupby(["district", "month"]).temp_mean.transform("mean")

    # monthly SPI / SPEI (z-score standardised per calendar month, per district)
    m = (df.assign(ym=df.date.dt.to_period("M").dt.to_timestamp())
         .groupby(["district", "ym"]).agg(rain=("rainfall_mm", "sum"), pet=("pet", "sum"),
                                          temp=("temp_mean", "mean"), days=("date", "count")).reset_index())
    m = m[m.days >= 25].copy()  # drop the partial current month
    m["wb"] = m.rain - m.pet
    m["month"], m["year"] = m.ym.dt.month, m.ym.dt.year
    for k in (1, 3, 6):
        for name, col in (("spi", "rain"), ("spei", "wb")):
            roll = m.groupby("district")[col].transform(lambda s: s.rolling(k).sum())
            m[f"{name}{k}"] = roll.groupby([m.district, m.month]).transform(lambda s: (s - s.mean()) / s.std())
    m["zone"] = m.district.map(ZONE)

    df.drop(columns=["lat", "lon"]).to_parquet("data/daily.parquet", index=False)
    m.to_parquet("data/monthly.parquet", index=False)
    print("daily", df.shape, "| monthly", m.shape, "| last date", df.date.max().date())


if __name__ == "__main__":
    main()
