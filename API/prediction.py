import pickle
import numpy as np
import pandas as pd
import os
from io import StringIO

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from sklearn.linear_model import LinearRegression, SGDRegressor
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
#  CORS Middleware
#  Configured specifically — NOT a wildcard (*)
#  Allows the Flutter mobile app and Render-hosted
#  frontend to call this API cross-origin securely.
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://localhost:3000",
        "https://your-app-name.onrender.com",   # ← replace with your Render URL after deployment
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
    """Load the best model and scaler from disk."""
    with open(os.path.join(BASE_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artefacts()

# ─────────────────────────────────────────────
#  Feature Engineering (mirrors the notebook exactly)
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "open",
    "high",
    "low",
    "intraday_range",
    "daily_return",
    "upper_wick",
    "lower_wick",
    "log_volume",
    "rolling_avg_5d",
    "open_close_ratio",
    "high_low_ratio",
]

def engineer_features(open_: float, high: float, low: float,
                       volume: float, rolling_avg_5d: float) -> pd.DataFrame:
    """
    Compute all engineered features from raw inputs.
    Must exactly mirror the feature engineering in multivariate.ipynb.
    """
    intraday_range   = high - low
    daily_return     = (open_ - open_) / open_ if open_ != 0 else 0.0  # at prediction time close≈open estimate
    upper_wick       = high - max(open_, open_)    # approximation at inference time
    lower_wick       = min(open_, open_) - low
    log_volume       = np.log1p(volume)
    open_close_ratio = 1.0                          # open/close ≈ 1 before we know close
    high_low_ratio   = high / low if low != 0 else 1.0

    features = {
        "open"            : open_,
        "high"            : high,
        "low"             : low,
        "intraday_range"  : intraday_range,
        "daily_return"    : daily_return,
        "upper_wick"      : upper_wick,
        "lower_wick"      : lower_wick,
        "log_volume"      : log_volume,
        "rolling_avg_5d"  : rolling_avg_5d,
        "open_close_ratio": open_close_ratio,
        "high_low_ratio"  : high_low_ratio,
    }
    return pd.DataFrame([features])[FEATURE_NAMES]

# ─────────────────────────────────────────────
#  Pydantic Input Schema — with types & range constraints
# ─────────────────────────────────────────────
class StockInput(BaseModel):
    open: float = Field(
        ...,
        gt=0,
        le=5000,
        description="Opening price of the stock in USD. Must be > 0 and ≤ 5000.",
        example=145.30
    )
    high: float = Field(
        ...,
        gt=0,
        le=5000,
        description="Highest price reached during the trading day in USD. Must be > 0 and ≤ 5000.",
        example=148.20
    )
    low: float = Field(
        ...,
        gt=0,
        le=5000,
        description="Lowest price reached during the trading day in USD. Must be > 0 and ≤ 5000.",
        example=144.10
    )
    volume: float = Field(
        ...,
        gt=0,
        le=10_000_000_000,
        description="Number of shares traded during the day. Must be > 0.",
        example=12_000_000
    )
    rolling_avg_5d: float = Field(
        ...,
        gt=0,
        le=5000,
        description="Average closing price over the past 5 trading days in USD. Must be > 0 and ≤ 5000.",
        example=144.80
    )

    @validator("high")
    def high_must_be_gte_low(cls, high, values):
        if "low" in values and high < values["low"]:
            raise ValueError("high must be greater than or equal to low")
        return high

    @validator("high")
    def high_must_be_gte_open(cls, high, values):
        if "open" in values and high < values["open"]:
            raise ValueError("high must be greater than or equal to open")
        return high

    @validator("low")
    def low_must_be_lte_open(cls, low, values):
        if "open" in values and low > values["open"]:
            raise ValueError("low must be less than or equal to open")
        return low

    class Config:
        schema_extra = {
            "example": {
                "open"          : 145.30,
                "high"          : 148.20,
                "low"           : 144.10,
                "volume"        : 12000000,
                "rolling_avg_5d": 144.80
            }
        }

# ─────────────────────────────────────────────
#  Pydantic Output Schema
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_close_price: float
    currency             : str = "USD"
    model_used           : str
    message              : str

class RetrainResponse(BaseModel):
    message         : str
    rows_used       : int
    new_test_mse    : float
    best_model      : str

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
    applies the saved StandardScaler, and returns the predicted close price.
    """
    try:
        # 1. Engineer features from raw inputs
        feature_df = engineer_features(
            open_         = data.open,
            high          = data.high,
            low           = data.low,
            volume        = data.volume,
            rolling_avg_5d= data.rolling_avg_5d,
        )

        # 2. Scale features using the saved scaler
        scaled = scaler.transform(feature_df)

        # 3. Predict
        prediction = float(model.predict(scaled)[0])

        # 4. Sanity check — predicted price must be positive
        if prediction <= 0:
            raise HTTPException(
                status_code=422,
                detail="Model returned an invalid prediction. Please check your input values."
            )

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
    Triggers model retraining when new data is uploaded.

    Upload a CSV file with columns: open, high, low, close, volume, rolling_avg_5d.
    The API will retrain all three models, select the best by Test MSE,
    and overwrite the saved best_model.pkl and scaler.pkl on disk.
    """
    global model, scaler

    # ── 1. Read uploaded CSV ──────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse the uploaded CSV file.")

    required_cols = {"open", "high", "low", "close", "volume", "rolling_avg_5d"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {missing}"
        )

    # ── 2. Clean uploaded data ────────────────────────────────────────────────
    df = df[list(required_cols)].dropna()
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) &
            (df["close"] > 0) & (df["volume"] > 0)]

    if len(df) < 50:
        raise HTTPException(
            status_code=400,
            detail="Not enough valid rows to retrain. Minimum 50 rows required."
        )

    # ── 3. Re-engineer features ───────────────────────────────────────────────
    df["intraday_range"]   = df["high"] - df["low"]
    df["daily_return"]     = (df["close"] - df["open"]) / df["open"]
    df["upper_wick"]       = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"]       = df[["open", "close"]].min(axis=1) - df["low"]
    df["log_volume"]       = np.log1p(df["volume"])
    df["open_close_ratio"] = df["open"] / df["close"]
    df["high_low_ratio"]   = df["high"] / df["low"]

    X = df[FEATURE_NAMES]
    y = df["close"]

    # ── 4. Split, scale ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    new_scaler = StandardScaler()
    X_train_s  = new_scaler.fit_transform(X_train)
    X_test_s   = new_scaler.transform(X_test)

    # ── 5. Train all three models & pick best ─────────────────────────────────
    candidates = {
        "LinearRegression"    : LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
    }

    best_name  = None
    best_mse   = float("inf")
    best_mdl   = None

    for name, mdl in candidates.items():
        mdl.fit(X_train_s, y_train)
        mse = mean_squared_error(y_test, mdl.predict(X_test_s))
        if mse < best_mse:
            best_mse  = mse
            best_name = name
            best_mdl  = mdl

    # ── 6. Overwrite saved artefacts ──────────────────────────────────────────
    with open(os.path.join(BASE_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_mdl, f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(new_scaler, f)

    # ── 7. Hot-swap in-memory model & scaler ─────────────────────────────────
    model  = best_mdl
    scaler = new_scaler

    return RetrainResponse(
        message     = "✅ Model retrained and updated successfully.",
        rows_used   = len(df),
        new_test_mse= round(best_mse, 4),
        best_model  = best_name,
    )
