import pandas as pd

from xgboost import XGBClassifier


def make_model():

    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )


def non_overlapping(df, horizon):

    parts = [
        g.sort_index().iloc[::max(horizon, 1)]
        for _, g in df.groupby("ticker")
    ]

    return pd.concat(parts) if parts else df