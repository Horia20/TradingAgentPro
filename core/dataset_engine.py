import pandas as pd


def non_overlapping(df, horizon):

    parts = [
        g.sort_index().iloc[::max(horizon, 1)]
        for _, g in df.groupby("ticker")
    ]

    return pd.concat(parts) if parts else df