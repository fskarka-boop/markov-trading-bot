import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path, header=None)
    df.columns = ["time","open","high","low","close","volume","close_time","quote_volume","trades","taker_volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def encode_states(df):
    df["return"] = df["close"].pct_change()
    df["state"] = pd.qcut(df["return"], 3, labels=[0,1,2])
    df["state"] = df["state"].astype(int)
    return df

def transition_matrix(states):
    n = 3
    P = np.zeros((n,n))
    for i in range(len(states)-1):
        P[states[i], states[i+1]] += 1
    P = P / P.sum(axis=1, keepdims=True)
    return P

def regime_filter(P):
    j_star = np.argmax(np.diag(P))
    p_hat = P[j_star, j_star]
    q = 1 - p_hat
    delta = p_hat - q
    stable = delta > 0
    return j_star, p_hat, q, delta, stable

def signal_from_regime(j_star, stable):
    if not stable:
        return "FLAT"
    if j_star == 2:
        return "LONG"
    if j_star == 0:
        return "SHORT"
    return "FLAT"
