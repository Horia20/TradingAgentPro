import pandas as pd

from core.ml_engine import make_model
from core.backtest_engine import ai_strategy_summary
from core.config import FEATURES


def walk_forward_validation(
    dataset,
    folds=5,
    threshold=0.40
):

    dates = sorted(
        dataset.index.unique()
    )

    results = []

    for i in range(folds):

        train_end = int(
            len(dates) *
            (0.50 + i * 0.10)
        )

        test_end = int(
            len(dates) *
            (0.60 + i * 0.10)
        )

        if test_end >= len(dates):
            break

        train_dates = dates[:train_end]
        test_dates = dates[train_end:test_end]

        train = dataset[
            dataset.index.isin(train_dates)
        ]

        test = dataset[
            dataset.index.isin(test_dates)
        ]

        model = make_model()

        model.fit(
            train[FEATURES],
            train["label"]
        )

        proba = model.predict_proba(
            test[FEATURES]
        )[:, 1]

        test = test.copy()

        test["probability"] = proba

        summary = ai_strategy_summary(
            test,
            threshold=threshold
        )

        summary["fold"] = i + 1

        results.append(summary)

    return pd.DataFrame(results)