import pandas as pd


def forward_return(close, horizon):

    return (
        close.shift(-horizon)
        / close
        - 1
    )


def win_rate(returns):

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return 0.0

    return float(
        (returns > 0).mean()
    )
def average_return(returns):

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return 0.0

    return float(
        returns.mean()
    )
def backtest_summary(returns):

    returns = pd.Series(returns).dropna()

    return {
        "trades": int(len(returns)),
        "win_rate": win_rate(returns),
        "average_return": average_return(returns),
        "best_trade": float(returns.max()) if len(returns) else 0.0,
        "worst_trade": float(returns.min()) if len(returns) else 0.0,
    }
def create_returns(close, horizon):

    r = forward_return(
        close,
        horizon
    )

    return r.dropna()
def buy_and_hold_backtest(close, horizon):

    returns = create_returns(
        close,
        horizon
    )

    return backtest_summary(
        returns
    )