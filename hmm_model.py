import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df.columns = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "qav", "num_trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ]

    df["time"] = df["open_time"].apply(lambda x: x / 1000 if x > 10**12 else x)
    df["time"] = pd.to_datetime(df["time"], unit="ms")

    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df["return"] = df["close"].pct_change()
    df["vol"] = df["return"].rolling(10).std()
    df["trend"] = df["close"].diff()

    df = df.dropna()
    return df


def train_hmm(df: pd.DataFrame, n_states: int = 3):
    X = df[["return", "vol", "trend"]].values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )

    model.fit(X)
    hidden_states = model.predict(X)

    df["regime"] = hidden_states
    return df, model

