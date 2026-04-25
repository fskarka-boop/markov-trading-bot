import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#────────────────────────────────────────────
# LOAD DATA
#────────────────────────────────────────────
df = pd.read_csv("data/BTCUSDT_1h.csv", header=None)
df.columns = [
    "open_time","open","high","low","close","volume",
    "close_time","qav","num_trades","tbv","tqv","ignore"
]

df["time"] = pd.to_datetime(df["open_time"], unit="ms")
df = df[["time","open","high","low","close","volume"]].copy()

#────────────────────────────────────────────
# VWMA
#────────────────────────────────────────────
def vwma(series, volume, length):
    return (series * volume).rolling(length).sum() / volume.rolling(length).sum()

df["vwma20"] = vwma(df["close"], df["volume"], 20)
df["vwma50"] = vwma(df["close"], df["volume"], 50)
df["vwma200"] = vwma(df["close"], df["volume"], 200)

# Trend
df["trend_long"]  = df["vwma50"] > df["vwma200"]
df["trend_short"] = df["vwma50"] < df["vwma200"]

# Breakout
df["break_long"]  = (df["close"] > df["vwma20"]) & (df["close"].shift(1) <= df["vwma20"].shift(1))
df["break_short"] = (df["close"] < df["vwma20"]) & (df["close"].shift(1) >= df["vwma20"].shift(1))

#────────────────────────────────────────────
# VOLATILITY CLOUD (BB ∩ KC)
#────────────────────────────────────────────
df["basisBB"] = df["close"].rolling(20).mean()
df["devBB"]   = df["close"].rolling(20).std()
df["upperBB"] = df["basisBB"] + df["devBB"] * 2
df["lowerBB"] = df["basisBB"] - df["devBB"] * 2

df["emaKC"]   = df["close"].ewm(span=20).mean()
df["atrKC"]   = (df["high"] - df["low"]).rolling(20).mean()
df["upperKC"] = df["emaKC"] + df["atrKC"] * 1.5
df["lowerKC"] = df["emaKC"] - df["atrKC"] * 1.5

df["cloudUpper"] = df[["upperBB","upperKC"]].min(axis=1)
df["cloudLower"] = df[["lowerBB","lowerKC"]].max(axis=1)
df["in_cloud"]   = (df["close"] <= df["cloudUpper"]) & (df["close"] >= df["cloudLower"])

#────────────────────────────────────────────
# MARKET STRUCTURE HL / LH
#────────────────────────────────────────────
df["HL"] = df["low"] > df["low"].rolling(3).min().shift(1)
df["LH"] = df["high"] < df["high"].rolling(3).max().shift(1)

#────────────────────────────────────────────
# FINAL SIGNAL
#────────────────────────────────────────────
df["long"]  = df["trend_long"]  & df["break_long"]  & df["HL"] & (~df["in_cloud"])
df["short"] = df["trend_short"] & df["break_short"] & df["LH"] & (~df["in_cloud"])

df["signal"] = np.where(df["long"], 1, np.where(df["short"], -1, 0))

#────────────────────────────────────────────
# BACKTEST
#────────────────────────────────────────────
df["return"] = df["close"].pct_change()
df["strategy_return"] = df["signal"].shift(1) * df["return"]
df["equity"] = (1 + df["strategy_return"].fillna(0)).cumprod()

# Stats
print("Total return:", df["equity"].iloc[-1] - 1)
print("Sharpe:", df["strategy_return"].mean() / df["strategy_return"].std() * np.sqrt(365))
print("Max DD:", (df["equity"].cummax() - df["equity"]).max())

# Plot
plt.figure(figsize=(12,6))
plt.plot(df["time"], df["equity"])
plt.title("VWMA Trend + Breakout Strategy")
plt.grid(True)
plt.show()
