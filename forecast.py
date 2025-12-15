import pandas as pd
import numpy as np
import pmdarima as pm
from tqdm import tqdm

# ================= USER INPUT =================
DATA_PATH = "dengue_2016_2025combined.csv"     # input dataset
OUTPUT_PATH = "dengue_forecast_2026.csv"

FEATURES = [
    "Dengue_Cases",
    "Dengue_Deaths",
    "Temperature_C",
    "Humidity_pct",
    "Rainfall_mm"
]

FORECAST_YEAR = 2026
MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
# =============================================


# ---------- Load Data ----------
df = pd.read_csv(DATA_PATH)

# Ensure proper month ordering
df["Month"] = pd.Categorical(df["Month"], categories=MONTHS, ordered=True)
df = df.sort_values(["Region", "Year", "Month"])

regions = df["Region"].unique()
future_rows = []

# ---------- Forecast Function ----------
def forecast_series(series, steps=12):
    if len(series) < 24:
        return [series.mean()] * steps
    model = pm.auto_arima(
        series,
        seasonal=True,
        m=12,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model.predict(n_periods=steps)


# ---------- Forecasting ----------
print("🔮 Forecasting Dengue & Weather Data for 2026...\n")

for region in tqdm(regions):
    region_df = df[df["Region"] == region]

    forecasts = {}
    for feature in FEATURES:
        forecasts[feature] = forecast_series(region_df[feature].values)

    for i, month in enumerate(MONTHS):
        row = {
            "Month": month,
            "Year": FORECAST_YEAR,
            "Region": region
        }

        for feature in FEATURES:
            row[feature] = max(0, round(forecasts[feature][i], 2))

        future_rows.append(row)


# ---------- Save Output ----------
forecast_df = pd.DataFrame(future_rows)
forecast_df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Forecasting Completed Successfully!")
print(f"📁 Output file saved as: {OUTPUT_PATH}")
