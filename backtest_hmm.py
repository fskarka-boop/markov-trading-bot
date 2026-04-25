import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hmm_model import load_data, prepare_features, train_hmm

DATA_PATH = "data/BTCUSDT_1h.csv"

df = load_data(DATA_PATH)
df = prepare_features(df)
df, model = train_hmm(df, n_states=3)

# --- Trend filtry ---
df["EMA50"] = df["close"].ewm(span=50).mean()
df["EMA200"] = df["close"].ewm(span=200).mean()

# --- ATR ---
df["H-L"] = df["high"] - df["low"]
df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
df["ATR"] = df["TR"].rolling(14).mean()
atr_threshold = df["ATR"].quantile(0.25)

# --- Mapování režimů ---
# Najdeme průměrné výnosy v každém režimu
regime_returns = df.groupby("regime")["return"].mean().sort_values()

bear_regime = regime_returns.index[0]
chop_regime = regime_returns.index[1]
bull_regime = regime_returns.index[2]

def map_regime_to_signal(row):
    if row["regime"] == bull_regime:
        return "LONG"
    if row["regime"] == bear_regime:
        return "SHORT"
    return "FLAT"

df["raw_signal"] = df.apply(map_regime_to_signal, axis=1)

# --- Filtry ---
def apply_filters(row):
    if pd.isna(row["ATR"]) or row["ATR"] < atr_threshold:
        return "FLAT"

    if row["raw_signal"] == "LONG" and row["EMA50"] < row["EMA200"]:
        return "FLAT"

    if row["raw_signal"] == "SHORT" and row["EMA50"] > row["EMA200"]:
        return "FLAT"

    return row["raw_signal"]

df["signal"] = df.apply(apply_filters, axis=1)

# --- PnL ---
df["position"] = df["signal"].replace({"LONG": 1, "SHORT": -1, "FLAT": 0})
df["strategy_return"] = df["position"].shift(1) * df["return"]
df["equity"] = (1 + df["strategy_return"].fillna(0)).cumprod()

# --- Statistika ---
print(df[["time", "close", "signal"]].tail(20))

print("\n===== STATISTIKY =====")
total_ret = df["equity"].iloc[-1] - 1
annual_ret = df["equity"].iloc[-1] ** (365 / len(df)) - 1
sharpe = df["strategy_return"].mean() / df["strategy_return"].std() * (365 ** 0.5) if df["strategy_return"].std() != 0 else 0
max_dd = (df["equity"].cummax() - df["equity"]).max()
trades = (df["position"].diff().abs() > 0).sum()

print("Celkový výnos:", round(total_ret, 4))
print("Roční výnos:", round(annual_ret, 4))
print("Sharpe:", round(sharpe, 4))
print("Max drawdown:", round(max_dd, 4))
print("Počet obchodů:", trades)

# --- Equity curve ---
plt.figure(figsize=(12, 6))
plt.plot(df["time"], df["equity"])
plt.title("Equity Curve – HMM 3-Regime Strategy")
plt.xlabel("Time")
plt.ylabel("Equity")
plt.grid(True)
plt.tight_layout()
plt.show()

