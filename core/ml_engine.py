import pandas as pd

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from core.indicators import add_features
from core.config import FEATURES


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


def build_dataset(data, horizon, target):

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
                FEATURES + ["fwd", "ticker"]
            ]
        )

    if not parts:
        return pd.DataFrame()

    ds = pd.concat(parts)

    ds["label"] = (
        ds["fwd"] >= target
    ).astype(int)

    return ds


def train_model(data, horizon=10, target=0.05):

    ds = build_dataset(
        data=data,
        horizon=horizon,
        target=target
    )

    if len(ds) < 500:
        return None

    dates = sorted(ds.index.unique())

    cut = dates[int(len(dates) * 0.75)]

    train = ds[ds.index <= cut]
    test = ds[ds.index > cut]

    if len(test) < 50:
        return None

    model = make_model()

    model.fit(
        train[FEATURES],
        train["label"]
    )

    proba = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    auc = roc_auc_score(
        test["label"],
        proba
    )

    metrics = {
        "auc": float(auc),
        "n_train": len(train),
        "n_test": len(test),
    }

    return {
        "model": model,
        "metrics": metrics,
    }
