**S&P 500 Stock Closing Price Prediction**

**Mission**
To empower retail investors and financial analysts with a data-driven model that predicts the daily closing price of S&P 500 stocks, enabling smarter, evidence-based investment decisions using historical trading signals.
Problem
Stock markets generate massive volumes of daily trading data that most retail investors cannot meaningfully interpret. This project builds a machine learning regression model that predicts a stock's daily closing price from its trading activity — open price, daily high, daily low, and volume — enriched with engineered financial indicators such as intraday volatility, daily return, and a 5-day rolling price average.

**Dataset**
Source: S&P 500 Stock Data — Kaggle (camnugent/sandp500)
Size: ~619,000 rows × 7 columns — daily trading records for 500+ companies over 5 years (2013–2018)
Features: Date, Open, High, Low, Close, Volume, Stock Ticker (Name)
Target Variable: close — the daily closing price of a stock (continuous, regression target)

**Visualizations**
Two key visualizations drive the feature engineering and model decisions:
1. Correlation Heatmap
Shows that open, high, low, and rolling_avg_5d correlate above 0.99 with the target close — confirming they are the strongest predictors. log_volume and daily_return show near-zero correlation, flagging them as weak standalone predictors.
2. Feature Scatter Plots (Open, Rolling Avg, Intraday Range, Log Volume vs Close)
Visually confirms the near-perfect linear relationship between open and close, the strong trend signal from rolling_avg_5d, and the weak/noisy signal from log_volume — directly informing which features to prioritize in the model.

**Repository Structure**
linear_regression_model/
│
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb        ← Full notebook: EDA, training, evaluation
│   ├── API/
│   │   ├── prediction.py             ← FastAPI prediction & retraining endpoints
│   │   ├── requirements.txt          ← Python dependencies
│   ├── FlutterApp/                   ← Mobile app source code
│
├── README.md
