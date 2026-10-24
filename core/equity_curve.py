import pandas as pd


def equity_curve(
    returns,
    initial_capital=10000
):

    equity = [initial_capital]

    capital = initial_capital

    for r in returns:

        capital *= (1 + r)

        equity.append(capital)

    return pd.DataFrame(
        {
            "step": range(len(equity)),
            "capital": equity
        }
    )