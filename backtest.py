import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from markov_model import load_data, encode_states, regime_filter, signal_from_regime

# cesta k datům
DATA_PATH = "data/BTCUSDT_1h.csv"

# načtení a příprava
df = load_data(DATA_PATH)
df = encode_states(df)

# --- EMA50 ---
df["EMA50"] = df["close"].ewm(span=50).mean()

# --- ATR ---
df["H-L"] = df["high"] - df["low"]
df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
df["ATR"] = df["TR"].rolling(14).mean()
atr_threshold = df["ATR"].quantile(0.25)

def apply_filters(row, raw_signal):
    # ATR filter – chop zóna
    if pd.isna(row["ATR"]) or row["ATR"] < atr_threshold:
        return "FLAT"

    # EMA trend filter
    if raw_signal == "LONG" and row["close"] < row["EMA50"]:
        return "FLAT"
    if raw_signal == "SHORT" and row["close"] > row["EMA50"]:
        return "FLAT"

    return raw_signal

# --- Markov signály přes rolling okno ---
window = 100
signals = []

states = df["state"].values

for i in range(len(df)):
    if i < window:
        signals.append("FLAT")
        continue

    window_states = states[i - window:i]
    j_star, stable = regime_filter(window_states)
    raw_sig = signal_from_regime(j_star, stable)
    signals.append(raw_sig)

df["signal"] = signals

# aplikace EMA/ATR filtrů
df["raw_signal"] = df["signal"]
df["signal"] = df.apply(lambda r: apply_filters(r, r["raw_signal"]), axis=1)

# --- PnL výpočet ---
df["position"] = df["signal"].replace({"LONG": 1, "SHORT": -1, "FLAT": 0})
df["return"] = df["close"].pct_change()
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
plt.title("Equity Curve – Markov Strategy (EMA+ATR filtered)")
plt.xlabel("Time")
plt.ylabel("Equity")
plt.grid(True)
plt.tight_layout()
plt.show()
