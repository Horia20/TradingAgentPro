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


def add_features(df):

    d = df.copy()

    c = d["Close"]

    d["SMA20"] = c.rolling(20).mean()
    d["SMA50"] = c.rolling(50).mean()
    d["SMA100"] = c.rolling(100).mean()
    d["SMA200"] = c.rolling(200).mean()

    d["RSI14"] = rsi(c)

    d["ATR14"] = atr(d)

    d["ret_5"] = c.pct_change(5)
    d["ret_20"] = c.pct_change(20)
    d["ret_60"] = c.pct_change(60)

    d["dist_sma50"] = c / d["SMA50"] - 1
    d["dist_sma200"] = c / d["SMA200"] - 1

    d["sma50_200"] = d["SMA50"] / d["SMA200"] - 1

    d["atr_pct"] = d["ATR14"] / c

    d["vol_ratio"] = (
        d["Volume"]
        / d["Volume"].rolling(20).mean()
    ).fillna(1.0)

    d["dist_high"] = (
        c / c.rolling(252).max() - 1
    )

    d["volat_20"] = (
        c.pct_change().rolling(20).std()
    )

    ema12 = c.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = c.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    d["macd_hist"] = macd - signal

    ll = d["Low"].rolling(14).min()
    hh = d["High"].rolling(14).max()

    k = 100 * (
        c - ll
    ) / (
        hh - ll
    ).replace(0, np.nan)

    d["stoch_k"] = k.rolling(3).mean()

    d["roc12"] = c.pct_change(12) * 100

    tp = (
        d["High"]
        + d["Low"]
        + c
    ) / 3

    d["cci20"] = (
        tp - tp.rolling(20).mean()
    ) / (
        0.015 *
        tp.rolling(20).apply(
            lambda x: np.abs(
                x - x.mean()
            ).mean(),
            raw=True
        )
    )

    return d.replace(
        [np.inf, -np.inf],
        np.nan
    )