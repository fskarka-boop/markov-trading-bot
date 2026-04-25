import numpy as np
import pandas as pd

# --- 1) Načtení dat ---
def load_data(path):
    df = pd.read_csv(path)
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()
    return df

# --- 2) Diskretizace do stavů ---
def encode_states(df):
    q = df["ret"].quantile([0.1, 0.3, 0.7, 0.9])
    q1, q2, q3, q4 = q[0.1], q[0.3], q[0.7], q[0.9]

    def enc(r):
        if r <= q1: return 0
        elif r <= q2: return 1
        elif r <= q3: return 2
        elif r <= q4: return 3
        else: return 4

    df["state"] = df["ret"].apply(enc)
    return df

# --- 3) Přechodová matice ---
def transition_matrix(states, n_states=5):
    counts = np.zeros((n_states, n_states))
    for i in range(len(states)-1):
        counts[states[i], states[i+1]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums

# --- 4) Tvoje podmínky: Δ a stabilita ---
def regime_filter(P, delta_min=0.05, p_min=0.87):
    self_probs = np.diag(P)
    j_star = int(np.argmax(self_probs))
    p_hat = self_probs[j_star]

    others = np.delete(self_probs, j_star)
    q = float(np.max(others)) if len(others) > 0 else 0.0

    delta = p_hat - q
    stable = (delta >= delta_min) and (p_hat >= p_min)

    return j_star, p_hat, q, delta, stable

# --- 5) Signál ---
def signal_from_regime(j_star, stable):
    if not stable:
        return "FLAT"
    if j_star in [3, 4]:
        return "LONG"
    if j_star in [0, 1]:
        return "SHORT"
    return "FLAT"

