import pandas as pd


def equal_weight_portfolio(returns):

    returns = pd.Series(
        returns
    ).dropna()

    capital = 10000.0

    if len(returns) == 0:

        return {
            "trades": 0,
            "average_return": 0.0,
            "capital": capital,
        }

    for r in returns:

        capital *= (
            1 + r
        )

    return {
        "trades": int(len(returns)),
        "average_return": float(
            returns.mean()
        ),
        "capital": float(capital),
    }