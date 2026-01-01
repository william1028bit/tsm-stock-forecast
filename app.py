# app.py
import os
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import yfinance as yf
import ta

from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Stock Forecast (Next 5–10 Days)", layout="wide")


# -----------------------------
# Utils
# -----------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@st.cache_resource
def load_bundle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


@st.cache_data(ttl=60 * 60)
def fetch_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise ValueError("No data returned. Check ticker/start date.")
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a superset of features. We'll later align to feature_cols from joblib.
    This keeps app robust if your training feature set changes.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # TA features (lots)
    df = ta.add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )

    # Custom, high-signal features (similar spirit to your v2)
    df["month"] = df["Date"].dt.month
    df["dayofweek"] = df["Date"].dt.dayofweek

    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)
    df["logret_1"] = np.log(df["Close"]).diff(1)

    # rolling stats / RSI-ish
    for w in [5, 10, 20]:
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()
        df[f"std_{w}"] = df["Close"].rolling(w).std()
        up = df["Close"].diff().clip(lower=0).rolling(w).mean()
        down = df["Close"].diff().clip(upper=0).abs().rolling(w).mean()
        df[f"rsi_{w}"] = 100 - (100 / (1 + (up / (down + 1e-9))))

    for lag in [1, 2, 3, 5]:
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)
        df[f"open_lag_{lag}"] = df["Open"].shift(lag)
        df[f"vol_lag_{lag}"] = df["Volume"].shift(lag)

    df["hl_range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["oc_range"] = (df["Open"] - df["Close"]) / (df["Close"] + 1e-9)

    # Some rows will be NaN due to rolling/lag; keep them, we will drop later
    return df


def next_trading_days(start_date: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    """
    Prefer NYSE calendar (handles holidays). Fallback to business days if package missing.
    """
    start_date = pd.to_datetime(start_date).normalize()

    # Try pandas_market_calendars if available
    try:
        import pandas_market_calendars as mcal  # type: ignore

        nyse = mcal.get_calendar("NYSE")
        # generate a window big enough
        end_guess = start_date + pd.Timedelta(days=30 + n * 3)
        sched = nyse.schedule(start_date=start_date, end_date=end_guess)
        days = list(pd.to_datetime(sched.index).normalize())
        # We want days strictly AFTER start_date
        days = [d for d in days if d > start_date]
        return days[:n]
    except Exception:
        # Fallback: business days (may include market holidays)
        bdays = pd.bdate_range(start_date + pd.Timedelta(days=1), periods=n, freq="B")
        return list(pd.to_datetime(bdays).normalize())


def model_predict_logret(model, X_row: pd.DataFrame) -> float:
    return float(model.predict(X_row)[0])


def align_X(df_feat: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # exact order, fill missing
    return df_feat.reindex(columns=feature_cols, fill_value=0.0)


# -----------------------------
# UI - Sidebar
# -----------------------------
st.title("Stock Forecast — Next 5–10 Trading Days (XGBoost log-return)")

with st.sidebar:
    st.header("Model & Data")

    model_path = st.text_input("Model file (joblib)", value="tsm_open_model.joblib")
    ticker = st.text_input("Ticker", value="TSM")
    start_date = st.text_input("History start", value="2015-01-01")

    st.divider()
    st.header("Forecast")

    horizon = st.slider("Forecast horizon (trading days)", min_value=5, max_value=10, value=5, step=1)
    price_basis = st.selectbox("Predict based on", ["Open", "Close"], index=0)

    st.divider()
    st.header("Backtest view")
    backtest_start = st.text_input("Backtest start (YYYY-MM-DD)", value="2024-01-01")
    show_debug = st.checkbox("Show debug table", value=False)


# -----------------------------
# Load bundle
# -----------------------------
try:
    bundle = load_bundle(model_path)
except Exception as e:
    st.error(f"❌ Cannot load model bundle: {e}")
    st.stop()

if "model" not in bundle or "feature_cols" not in bundle:
    st.error("❌ joblib bundle must contain keys: 'model', 'feature_cols'.")
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]
best_params = bundle.get("best_params", {})

st.caption(f"Loaded model from `{model_path}` | features: {len(feature_cols)} | best_params: {best_params}")


# -----------------------------
# Fetch & feature-engineer
# -----------------------------
try:
    df_raw = fetch_ohlcv(ticker, start_date)
except Exception as e:
    st.error(f"❌ Data download error: {e}")
    st.stop()

df_feat = add_features(df_raw)

# Target definition is model-specific; for backtest we only need true next-day price.
# Remove rows that cannot support feature creation (lags/rolling)
df_feat = df_feat.dropna().copy()

if len(df_feat) < 300:
    st.warning("Data length is short after feature engineering. Consider earlier start date.")
    # still continue


# -----------------------------
# 1) Single-step forecast (next trading day)
# -----------------------------
latest_row = df_feat.iloc[-1].copy()
today_price = float(latest_row[price_basis])

X_latest = align_X(df_feat, feature_cols).iloc[[-1]]
pred_logret_1 = model_predict_logret(model, X_latest)
pred_next_price_1 = today_price * math.exp(pred_logret_1)
pred_pct_1 = (pred_next_price_1 / today_price - 1) * 100


# KPI cards
c1, c2, c3 = st.columns(3)
c1.metric(f"Today {price_basis}", f"{today_price:.2f}")
c2.metric(f"Predicted next-day {price_basis}", f"{pred_next_price_1:.2f}")
c3.metric("Predicted % change (next day)", f"{pred_pct_1:.2f}%")


# -----------------------------
# 2) Multi-step forecast (5–10 days) — recursive + recompute features
# -----------------------------
st.subheader(f"Next {horizon} trading days forecast ({price_basis})")

# Use a recent window for speed but enough to compute indicators
WINDOW = 300
base_raw = df_raw.copy()
base_raw["Date"] = pd.to_datetime(base_raw["Date"])

# Keep last WINDOW rows for simulation
base_raw_sim = base_raw.tail(WINDOW).copy().reset_index(drop=True)
last_known_date = pd.to_datetime(base_raw_sim["Date"].iloc[-1]).normalize()

future_dates = next_trading_days(last_known_date, horizon)

# If user wants to start from a provided price (like 303.89), let them override quickly
with st.expander("Optional: override today's price (for what-if)", expanded=False):
    override = st.checkbox("Use manual price as starting point", value=False)
    manual_price = st.number_input("Manual starting price", value=float(today_price), step=0.01)

start_price = float(manual_price) if override else float(today_price)

# Ensure last row basis price equals start_price (what-if)
base_raw_sim.loc[base_raw_sim.index[-1], price_basis] = start_price
# Keep OHLC consistent for what-if (simple but stable)
base_raw_sim.loc[base_raw_sim.index[-1], "Open"] = start_price
base_raw_sim.loc[base_raw_sim.index[-1], "High"] = start_price
base_raw_sim.loc[base_raw_sim.index[-1], "Low"] = start_price
base_raw_sim.loc[base_raw_sim.index[-1], "Close"] = start_price
if "Adj Close" in base_raw_sim.columns:
    base_raw_sim.loc[base_raw_sim.index[-1], "Adj Close"] = start_price


pred_records = []
curr_price = start_price

# Iterative forecast
for i, d in enumerate(future_dates, start=1):
    # Recompute features on current simulated history
    feat_sim = add_features(base_raw_sim).dropna().copy()

    X_sim = align_X(feat_sim, feature_cols)
    X_last = X_sim.iloc[[-1]]

    pred_lr = model_predict_logret(model, X_last)
    next_price = curr_price * math.exp(pred_lr)
    pct = (next_price / curr_price - 1) * 100

    pred_records.append(
        {"Date": d, f"Pred_{price_basis}": next_price, "Pct_Change": pct}
    )

    # Append synthetic row for next day (keep volume as last known; OHLC set to predicted)
    new_row = base_raw_sim.iloc[[-1]].copy()
    new_row.loc[new_row.index[0], "Date"] = pd.to_datetime(d)
    new_row.loc[new_row.index[0], "Open"] = next_price
    new_row.loc[new_row.index[0], "High"] = next_price
    new_row.loc[new_row.index[0], "Low"] = next_price
    new_row.loc[new_row.index[0], "Close"] = next_price
    if "Adj Close" in new_row.columns:
        new_row.loc[new_row.index[0], "Adj Close"] = next_price

    base_raw_sim = pd.concat([base_raw_sim, new_row], ignore_index=True)

    curr_price = next_price

forecast_df = pd.DataFrame(pred_records)
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

# Show headline "tomorrow"
tomorrow_row = forecast_df.iloc[0]
st.info(
    f"Tomorrow predicted {price_basis}: **{tomorrow_row[f'Pred_{price_basis}']:.2f}** "
    f"({tomorrow_row['Pct_Change']:+.2f}%)  "
    f"starting from {price_basis}={start_price:.2f}"
)

# Plot: predicted path
fig1 = plt.figure(figsize=(10, 4))
plt.plot(forecast_df["Date"], forecast_df[f"Pred_{price_basis}"], marker="o", linewidth=1)
plt.title(f"Predicted {price_basis} Path (Next {horizon} Trading Days)")
plt.xlabel("Date")
plt.ylabel(f"Predicted {price_basis}")
plt.grid(True)
plt.tight_layout()
st.pyplot(fig1)

# Plot: bar pct change
fig2 = plt.figure(figsize=(10, 3))
plt.bar(forecast_df["Date"].dt.strftime("%m-%d"), forecast_df["Pct_Change"])
plt.title("Predicted Daily % Change")
plt.xlabel("Day")
plt.ylabel("%")
plt.grid(True, axis="y")
plt.tight_layout()
st.pyplot(fig2)

# Table
st.dataframe(
    forecast_df.assign(
        **{
            f"Pred_{price_basis}": forecast_df[f"Pred_{price_basis}"].map(lambda x: round(float(x), 3)),
            "Pct_Change": forecast_df["Pct_Change"].map(lambda x: round(float(x), 3)),
        }
    )
)


# -----------------------------
# 3) Backtest chart (intuitive)
# -----------------------------
st.subheader("Backtest (Actual vs Predicted next-day)")

bt_start = pd.to_datetime(backtest_start)
df_bt = df_feat[df_feat["Date"] >= bt_start].copy()
df_bt = df_bt.dropna().copy()

X_bt = align_X(df_bt, feature_cols)
pred_lr_bt = model.predict(X_bt)

today_bt = df_bt[price_basis].values
pred_next_bt = today_bt * np.exp(pred_lr_bt)

true_next_bt = df_bt[price_basis].shift(-1).values

# align
pred_next_bt = pred_next_bt[:-1]
true_next_bt = true_next_bt[:-1]
dates_bt = df_bt["Date"].values[:-1]

mae_bt = mean_absolute_error(true_next_bt, pred_next_bt)
rmse_bt = rmse(true_next_bt, pred_next_bt)

k1, k2 = st.columns(2)
k1.metric("Backtest MAE", f"{mae_bt:.3f}")
k2.metric("Backtest RMSE", f"{rmse_bt:.3f}")

fig3 = plt.figure(figsize=(12, 5))
plt.plot(dates_bt, true_next_bt, label="Actual next-day", linewidth=1)
plt.plot(dates_bt, pred_next_bt, label="Predicted next-day", linewidth=1)
plt.title(f"{ticker} — Actual vs Predicted next-day {price_basis} (from {backtest_start})")
plt.xlabel("Date")
plt.ylabel(price_basis)
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig3)


# -----------------------------
# 4) Debug (optional)
# -----------------------------
if show_debug:
    st.subheader("Latest rows (debug)")
    dbg = df_feat[["Date", "Open", "Close"]].copy().tail(15)
    dbg[f"pred_next_{price_basis.lower()}"] = (df_feat[price_basis].values * np.exp(model.predict(align_X(df_feat, feature_cols)))).round(4)
    st.dataframe(dbg.tail(10))
