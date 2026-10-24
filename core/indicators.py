import numpy as np
import pandas as pd


def rsi(close, n=14):
    d = close.diff()

    up = d.clip(lower=0).ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    dn = (-d.clip(upper=0)).ewm(
        alpha=1 / n,
        adjust=False
    ).mean()

    return 100 - 100 / (
        1 + up / dn.replace(0, np.nan)
    )


def atr(df, n=14):

    pc = df["Close"].shift()

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - pc).abs(),
            (df["Low"] - pc).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()