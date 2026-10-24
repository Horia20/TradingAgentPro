import pandas as pd

from core.indicators import add_features
from core.config import FEATURES


def non_overlapping(df, horizon):

    parts = [
        g.sort_index().iloc[::max(horizon, 1)]
        for _, g in df.groupby("ticker")
    ]

    return pd.concat(parts) if parts else df


def build_dataset(
    data,
    horizon,
    target
):

    parts = []

    for ticker, df in data.items():

        f = add_features(df)

        f["fwd"] = (
            f["Close"].shift(-horizon)
            / f["Close"]
            - 1
        )

        f["ticker"] = ticker

        parts.append(
            f.dropna(
                subset=FEATURES + ["fwd"]
            )[
                FEATURES + [
                    "fwd",
                    "ticker"
                ]
            ]
        )

    if not parts:
        return pd.DataFrame()

    ds = pd.concat(parts)

    ds["label"] = (
        ds["fwd"] >= target
    ).astype(int)

    return ds