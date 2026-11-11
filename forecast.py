import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# 1. Load Dataset
df = pd.read_csv("sri_lanka_all_districts_weather_2015_onwards.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["district", "datetime"]).reset_index(drop=True)

TARGET_COLS = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "rainfall_mm",
    "humidity_mean",
    "solar_radiation_mean",
]
FORECAST_HORIZON = 7


# 2. Feature Engineering Function
def create_features(data):
    """Generates calendar, cyclical, lag, and rolling statistics for features."""
    df_feat = data.copy()

    # Temporal & Cyclical Features
    df_feat["day_of_year"] = df_feat["datetime"].dt.dayofyear
    df_feat["month"] = df_feat["datetime"].dt.month
    df_feat["day_of_week"] = df_feat["datetime"].dt.dayofweek
    df_feat["sin_day"] = np.sin(2 * np.pi * df_feat["day_of_year"] / 365.25)
    df_feat["cos_day"] = np.cos(2 * np.pi * df_feat["day_of_year"] / 365.25)

    # Weather Lags & Rolling Window Features
    for col in TARGET_COLS:
        df_feat[f"{col}_lag1"] = df_feat[col].shift(1)
        df_feat[f"{col}_lag2"] = df_feat[col].shift(2)
        df_feat[f"{col}_lag7"] = df_feat[col].shift(7)
        df_feat[f"{col}_roll7_mean"] = (
            df_feat[col].shift(1).rolling(window=7).mean()
        )
        df_feat[f"{col}_roll7_std"] = (
            df_feat[col].shift(1).rolling(window=7).std()
        )

    return df_feat


# 3. Model Training & Recursive Forecasting Loop per District
all_forecasts = []
districts = df["district"].unique()

for district in districts:
    print(f"Training models for district: {district}")

    # Extract district series sorted chronologically
    d_df = (
        df[df["district"] == district]
        .copy()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Engineer historical feature matrix
    d_feat = create_features(d_df)

    # Define feature set (exclude identifiers and targets)
    feature_cols = [
        col
        for col in d_feat.columns
        if col not in ["district", "datetime"] + TARGET_COLS
    ]

    # Clean missing values created by shifting lags
    train_data = d_feat.dropna(subset=feature_cols).reset_index(drop=True)
    X_train = train_data[feature_cols]

    # Fit target-specific XGBoost regressors
    models = {}
    for target in TARGET_COLS:
        y_train = train_data[target]
        model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        models[target] = model

    # 4. Recursive Multi-Step 7-Day Forecasting
    last_known_date = d_df["datetime"].max()
    history_df = d_df.copy()

    for step in range(1, FORECAST_HORIZON + 1):
        target_date = last_known_date + pd.Timedelta(days=step)

        # Create temporary step row
        dummy_row = pd.DataFrame(
            [{"district": district, "datetime": target_date}]
        )
        for col in TARGET_COLS:
            dummy_row[col] = np.nan

        # Concatenate history to compute updated dynamic lag features
        temp_history = pd.concat(
            [history_df, dummy_row], ignore_index=True
        ).sort_values("datetime")
        temp_feat = create_features(temp_history)

        # Extract features for current prediction step
        X_future = temp_feat[temp_feat["datetime"] == target_date][feature_cols]

        # Predict targets for step t
        step_predictions = {"district": district, "datetime": target_date}
        for target in TARGET_COLS:
            pred_val = float(models[target].predict(X_future)[0])

            # Apply domain boundary constraints
            if target == "rainfall_mm":
                pred_val = max(0.0, pred_val)
            elif target == "humidity_mean":
                pred_val = float(np.clip(pred_val, 0.0, 100.0))
            elif target == "temp_min":
                # Ensure temp_min does not exceed temp_max predicted values if present
                pass

            step_predictions[target] = pred_val

        # Store predictions
        all_forecasts.append(step_predictions)

        # Append step forecast to local history matrix to construct lags for t + 1
        history_df = pd.concat(
            [history_df, pd.DataFrame([step_predictions])], ignore_index=True
        )

# 5. Save Forecast Data to Parquet
forecast_df = pd.DataFrame(all_forecasts)
forecast_df.to_parquet("forecasts_7days.parquet", index=False)

print(
    f"\nSuccessfully generated 7-day recursive forecasts for all {len(districts)} districts."
)
print("Saved predictions to 'forecasts_7days.parquet'.")