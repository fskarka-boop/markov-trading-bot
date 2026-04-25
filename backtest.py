from markov_model import *
import pandas as pd

df = load_data("data/BTCUSDT_1h.csv")
df = encode_states(df)

signals = []
window = 50  # rolling okno (cca 3 týdny)

for i in range(window, len(df)):
    window_states = df["state"].iloc[i-window:i].values
    P = transition_matrix(window_states)

    j_star, p_hat, q, delta, stable = regime_filter(P)
    sig = signal_from_regime(j_star, stable)

    signals.append(sig)

df = df.iloc[window:].copy()
df["signal"] = signals

print(df[["time", "close", "signal"]].tail(20))

# Výpočet PnL
df["position"] = df["signal"].replace({"LONG": 1, "SHORT": -1, "FLAT": 0})
df["return"] = df["close"].pct_change()
df["strategy_return"] = df["position"].shift(1) * df["return"]

df["equity"] = (1 + df["strategy_return"].fillna(0)).cumprod()

print("\n===== STATISTIKY =====")
print("Celkový výnos:", round(df["equity"].iloc[-1] - 1, 4))
print("Roční výnos:", round((df["equity"].iloc[-1] ** (365/len(df)) - 1), 4))
print("Sharpe:", round(df["strategy_return"].mean() / df["strategy_return"].std() * (365**0.5), 4))
print("Max drawdown:", round((df["equity"].cummax() - df["equity"]).max(), 4))
print("Počet obchodů:", (df["position"].diff().abs() > 0).sum())



