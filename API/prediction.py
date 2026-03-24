import pickle
import numpy as np
import pandas as pd
import os
from io import StringIO
from typing import Annotated

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────
#  App Initialisation
# ─────────────────────────────────────────────
app = FastAPI(
    title="S&P 500 Stock Closing Price Predictor",
    description=(
        "Predicts the daily closing price of an S&P 500 stock "
        "from raw trading inputs (open, high, low, volume, rolling average). "
        "Built with Linear Regression, Decision Tree, and Random Forest models."
    ),
    version="1.0.0",
)

# ─────────────────────────────────────────────
#  CORS Middleware — specifically configured, NOT wildcard
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://localhost:3000",
        "https://stock-price-predictor-cetx.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# ─────────────────────────────────────────────
#  Load Saved Model Artefacts
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artefacts():
    with open(os.path.join(BASE_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artefacts()

# ─────────────────────────────────────────────
#  Feature Names & Engineering
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "open", "high", "low", "intraday_range", "daily_return",
    "upper_wick", "lower_wick", "log_volume", "rolling_avg_5d",
    "open_close_ratio", "high_low_ratio",
]

def engineer_features(
    open_: float, high: float, low: float,
    volume: float, rolling_avg_5d: float
) -> pd.DataFrame:
    features = {
        "open"            : open_,
        "high"            : high,
        "low"             : low,
        "intraday_range"  : high - low,
        "daily_return"    : 0.0,
        "upper_wick"      : high - open_,
        "lower_wick"      : open_ - low,
        "log_volume"      : np.log1p(volume),
        "rolling_avg_5d"  : rolling_avg_5d,
        "open_close_ratio": 1.0,
        "high_low_ratio"  : high / low if low != 0 else 1.0,
    }
    return pd.DataFrame([features])[FEATURE_NAMES]

# ─────────────────────────────────────────────
#  Pydantic v2 Schemas
# ─────────────────────────────────────────────
class StockInput(BaseModel):
    open: Annotated[float, Field(gt=0, le=5000, description="Opening price in USD", examples=[145.30])]
    high: Annotated[float, Field(gt=0, le=5000, description="Daily high price in USD", examples=[148.20])]
    low:  Annotated[float, Field(gt=0, le=5000, description="Daily low price in USD", examples=[144.10])]
    volume: Annotated[float, Field(gt=0, le=10_000_000_000, description="Shares traded", examples=[12000000])]
    rolling_avg_5d: Annotated[float, Field(gt=0, le=5000, description="5-day average close price in USD", examples=[144.80])]

    @model_validator(mode="after")
    def validate_price_relationships(self):
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.high < self.open:
            raise ValueError("high must be >= open")
        if self.low > self.open:
            raise ValueError("low must be <= open")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "open": 145.30, "high": 148.20, "low": 144.10,
                "volume": 12000000, "rolling_avg_5d": 144.80
            }]
        }
    }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_close_price: float
    currency             : str = "USD"
    model_used           : str
    message              : str


class RetrainResponse(BaseModel):
    message     : str
    rows_used   : int
    new_test_mse: float
    best_model  : str


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Confirms the API is live and running."""
    return {
        "status" : "✅ API is live",
        "model"  : type(model).__name__,
        "version": "1.0.0",
        "docs"   : "/docs"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: StockInput):
    """
    Predicts the daily closing price of an S&P 500 stock.
    Accepts 5 raw trading inputs, engineers features internally,
    scales them and returns the predicted close price.
    """
    try:
        feature_df = engineer_features(
            open_=data.open, high=data.high, low=data.low,
            volume=data.volume, rolling_avg_5d=data.rolling_avg_5d,
        )
        scaled     = scaler.transform(feature_df)
        prediction = float(model.predict(scaled)[0])

        if prediction <= 0:
            raise HTTPException(status_code=422, detail="Invalid prediction. Check input values.")

        return PredictionResponse(
            predicted_close_price=round(prediction, 4),
            currency="USD",
            model_used=type(model).__name__,
            message="Prediction successful."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/retrain", response_model=RetrainResponse, tags=["Retraining"])
async def retrain(file: UploadFile = File(...)):
    """
    Triggers model retraining when new data is uploaded as a CSV file.
    Required columns: open, high, low, close, volume, rolling_avg_5d.
    Retrains all three models, saves the best, and hot-swaps it in memory.
    """
    global model, scaler

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse the uploaded CSV.")

    required_cols = {"open", "high", "low", "close", "volume", "rolling_avg_5d"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    df = df[list(required_cols)].dropna()
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) &
            (df["close"] > 0) & (df["volume"] > 0)]

    if len(df) < 50:
        raise HTTPException(status_code=400, detail="Need at least 50 valid rows to retrain.")

    df["intraday_range"]   = df["high"] - df["low"]
    df["daily_return"]     = (df["close"] - df["open"]) / df["open"]
    df["upper_wick"]       = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"]       = df[["open", "close"]].min(axis=1) - df["low"]
    df["log_volume"]       = np.log1p(df["volume"])
    df["open_close_ratio"] = df["open"] / df["close"]
    df["high_low_ratio"]   = df["high"] / df["low"]

    X = df[FEATURE_NAMES]
    y = df["close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    new_scaler = StandardScaler()
    X_train_s  = new_scaler.fit_transform(X_train)
    X_test_s   = new_scaler.transform(X_test)

    candidates = {
        "LinearRegression"     : LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
    }

    best_name, best_mse, best_mdl = None, float("inf"), None
    for name, mdl in candidates.items():
        mdl.fit(X_train_s, y_train)
        mse = mean_squared_error(y_test, mdl.predict(X_test_s))
        if mse < best_mse:
            best_mse, best_name, best_mdl = mse, name, mdl

    with open(os.path.join(BASE_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_mdl, f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(new_scaler, f)

    model  = best_mdl
    scaler = new_scaler

    return RetrainResponse(
        message="✅ Model retrained and updated successfully.",
        rows_used=len(df),
        new_test_mse=round(best_mse, 4),
        best_model=best_name,
    )
