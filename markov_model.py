from markov_model import *
import pandas as pd

df = load_data("data/BTCUSDT_1h.csv")
df = encode_states(df)

signals = []
window = 500  # rolling okno (cca 3 týdny)

for i in range(window, len(df)):
    window_states = df["state"].iloc[i-window:i].values
    P = transition_matrix(window_states)

    j_star, p_hat, q, delta, stable = regime_filter(P)
    sig = signal_from_regime(j_star, stable)

    signals.append(sig)

df = df.iloc[window:].copy()
df["signal"] = signals

print(df[["time", "close", "signal"]].tail(20))

