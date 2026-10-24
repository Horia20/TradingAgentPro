import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier

from core.indicators import add_features
from core.config import FEATURES

from core.dataset_engine import (
    build_dataset,
    non_overlapping,
)

from core.calibration_engine import (
    calibrate_model,
)


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


def train_model(
    data,
    horizon=10,
    target=0.05
):

    ds = build_dataset(
        data=data,
        horizon=horizon,
        target=target
    )

    if len(ds) < 500:
        return None

    dates = sorted(
        ds.index.unique()
    )

    cut = dates[
        int(len(dates) * 0.75)
    ]

    train = ds[
        ds.index <= cut
    ]

    test = ds[
        ds.index > cut
    ]

    if len(test) < 50:
        return None

    base_model = make_model()

    calibrated = calibrate_model(
        base_model
    )

    calibrated.fit(
        train[FEATURES],
        train["label"]
    )

    proba = calibrated.predict_proba(
        test[FEATURES]
    )[:, 1]

    auc = roc_auc_score(
        test["label"],
        proba
    )

    pred = (
        proba >= 0.50
    ).astype(int)

    precision = precision_score(
        test["label"],
        pred,
        zero_division=0
    )

    recall = recall_score(
        test["label"],
        pred,
        zero_division=0
    )

    f1 = f1_score(
        test["label"],
        pred,
        zero_division=0
    )

    metrics = {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_train": len(train),
        "n_test": len(test),
    }

    test = test.copy()

    test["probability"] = proba

    return {
        "model": calibrated,
        "metrics": metrics,
        "test_set": test,
    }


def predict_stock(
    model,
    df
):

    f = add_features(df)

    last = (
        f[FEATURES]
        .dropna()
        .tail(1)
    )

    if len(last) == 0:
        return None

    prob = model.predict_proba(
        last
    )[0][1]

    return float(prob)