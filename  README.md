# TSM Stock Forecast (XGBoost, Log-Return)

This project predicts **TSMC (NYSE: TSM)** next-day and next 5–10 trading day prices using an **XGBoost regression model trained on log-returns**.

## Features
- XGBoost regression on log-returns
- Technical indicators + lag features
- Walk-forward backtesting
- Next-day and multi-day price forecasting
- Interactive Streamlit dashboard

## Model
- Target: log(Open_{t+1} / Open_t)
- Training data: 2015 → present
- Evaluation: Holdout + Walk-forward backtest

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
