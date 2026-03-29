# 📈 S&P 500 Stock Closing Price Prediction

## Mission
To empower retail investors and financial analysts with a data-driven model that predicts the daily closing price of S&P 500 stocks, enabling smarter, evidence-based investment decisions using historical trading signals. This project builds a regression model that predicts a stock's daily closing price from trading activity — open price, daily high, daily low, and volume — enriched with engineered financial indicators such as intraday volatility, daily return, and a 5-day rolling price average.

## Dataset
**Source:** [S&P 500 Stock Data — Kaggle (camnugent/sandp500)](https://www.kaggle.com/datasets/camnugent/sandp500)
**Size:** ~619,000 rows × 7 columns — daily trading records for 500+ companies over 5 years (2013–2018)
**Target Variable:** `close` — the daily closing price of a stock (continuous, regression target)

---

## 🌐 Live API

**Public Endpoint:** `https://stock-price-predictor-cetx.onrender.com`
**Swagger UI:** `https://stock-price-predictor-cetx.onrender.com/docs`

> ⚠️ The API is hosted on Render's free tier. If it has been inactive, the first request may take up to 50 seconds to wake up. Please wait and retry.

---

## 🎥 Video Demo

**YouTube:** `https://youtu.be/your-video-link`
---

## 📱 How to Run the Flutter App

### Prerequisites
- [Flutter SDK](https://flutter.dev/docs/get-started/install) installed
- Android Studio with an emulator set up **OR** a physical Android device with USB debugging enabled
- Git installed

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/dav1d001/linear_regression_model.git
```

**2. Navigate to the Flutter app**
```bash
cd linear_regression_model/FlutterApp
```

**3. Install dependencies**
```bash
flutter pub get
```

**4. Run the app**
```bash
flutter run
```

**5. Using the app**
- Enter the stock's Open Price, Daily High, Daily Low, Trading Volume, and 5-Day Rolling Average
- Tap **Predict**
- The predicted closing price will appear in the result card below

---

## 📊 Visualizations

**1. Correlation Heatmap**
Shows that `open`, `high`, `low`, and `rolling_avg_5d` correlate above 0.99 with the target `close` — confirming they are the strongest predictors. `log_volume` and `daily_return` show near-zero correlation, flagging them as weak standalone predictors.

**2. Feature Scatter Plots**
Visually confirms the near-perfect linear relationship between `open` and `close`, the strong trend signal from `rolling_avg_5d`, and the weak/noisy signal from `log_volume` — directly informing feature prioritization.

---

## 🗂️ Repository Structure

```
linear_regression_model/
│
│   ├── linear_regression/
│   │   ├── multivariate.ipynb        ← Full notebook: EDA, training, evaluation
│   ├── API/
│   │   ├── prediction.py             ← FastAPI prediction & retraining endpoints
│   │   ├── requirements.txt          ← Python dependencies
│   │   ├── best_model.pkl            ← Saved best model
│   │   ├── scaler.pkl                ← Saved StandardScaler
│   │   ├── feature_names.pkl         ← Saved feature list
│   ├── FlutterApp/                   ← Mobile app source code
│
├── README.md
```

---

## 🤖 Models Trained

| Model | Description |
|---|---|
| Linear Regression (OLS) | Closed-form solution, fast and interpretable |
| SGD Linear Regression | Gradient descent with loss curve tracking |
| Decision Tree Regressor | Non-linear splits, depth-constrained to prevent overfitting |
| Random Forest Regressor | Ensemble of 100 trees |

The best-performing model (lowest test MSE) is saved as `best_model.pkl` and served via the API.
