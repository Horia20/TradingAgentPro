import pandas as pd

from core.config import FEATURES


def feature_importance_report(model):

    values = model.feature_importances_

    rows = []

    for feature, importance in zip(
        FEATURES,
        values
    ):

        rows.append(
            {
                "feature": feature,
                "importance": float(importance)
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "importance",
        ascending=False
    )

    df["rank"] = range(
        1,
        len(df) + 1
    )

    return df[
        [
            "rank",
            "feature",
            "importance"
        ]
    ]