import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df.columns = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "qav", "num_trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ]

    # timestamp fix (mikro vs. milisekundy)
    df["time"] = df["open_time"].apply(lambda x: x / 1000 if x > 10**12 else x)
    df["time"] = pd.to_datetime(df["time"], unit="ms")

    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df

def encode_states(df: pd.DataFrame) -> pd.DataFrame:
    df["return"] = df["close"].pct_change()

    df["state"] = pd.qcut(
        df["return"],
        3,
        labels=[0, 1, 2],
        duplicates="drop"
    )

    df = df.dropna(subset=["return", "state"])
    df["state"] = df["state"].astype(int)
    return df

def build_transition_matrix(states: np.ndarray, n_states: int = 3) -> np.ndarray:
    P = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        P[states[i], states[i + 1]] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = P / row_sums
    return P

def regime_filter(states_window: np.ndarray) -> tuple[int, bool]:
    P = build_transition_matrix(states_window)
    diag = np.diag(P)
    j_star = int(np.argmax(diag))
    p_hat = diag[j_star]

    # mírně změkčená stabilita
    stable = p_hat > 0.33
    return j_star, stable

def signal_from_regime(j_star: int, stable: bool) -> str:
    if not stable:
        return "FLAT"
    if j_star == 2:
        return "LONG"
    if j_star == 0:
        return "SHORT"
    return "FLAT"
