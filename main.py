from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Dengue Risk Prediction API")

# Enable Flutter CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load Dataset
# -----------------------
DATASET_PATH = "dengue_final_dataset.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset file not found. Place dengue_final_dataset.csv in project folder.")

df = pd.read_csv(DATASET_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

required = {
    "month", "year", "region", "dengue_cases", "dengue_deaths",
    "temperature_c", "humidity_pct", "rainfall_mm",
    "cluster", "risk_score", "risk_category"
}

missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Dataset missing columns: {missing}")

print("DEBUG COLUMNS:", df.columns.tolist())

# Convert month names → numbers (Jan = 1 …)
month_to_num = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}

# Add numeric month column
df["month_num"] = df["month"].str.lower().map(month_to_num)


@app.get("/")
def home():
    return {"message": "Dengue Prediction API is running!"}


@app.post("/predict")
def predict(region: str, month: int, year: int):
    """Returns dengue risk for a region/month/year using dataset-based cluster."""

    # Validate month
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Month must be 1–12")

    # Filter dataset
    subset = df[
        (df["region"].str.lower() == region.lower()) &
        (df["month_num"] == month) &
        (df["year"] == year)
    ]

    if subset.empty:
        raise HTTPException(
            status_code=404,
            detail="No matching data found for this region/month/year."
        )

    row = subset.iloc[0]

    cluster = int(row["cluster"])
    risk = row["risk_category"]

    return {
        "region": region,
        "month": month,
        "year": year,
        "cluster": cluster,
        "risk_level": risk,
        "values_used": {
            "dengue_cases": float(row["dengue_cases"]),
            "dengue_deaths": float(row["dengue_deaths"]),
            "temperature": float(row["temperature_c"]),
            "humidity": float(row["humidity_pct"]),
            "rainfall": float(row["rainfall_mm"]),
            "risk_score": float(row["risk_score"])
        }
    }
