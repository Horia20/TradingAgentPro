import pandas as pd


def forward_return(close, horizon):

    return (
        close.shift(-horizon)
        / close
        - 1
    )


def create_returns(close, horizon):

    r = forward_return(
        close,
        horizon
    )

    return r.dropna()


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


def profit_factor(returns):

    returns = pd.Series(returns).dropna()

    gains = returns[returns > 0].sum()

    losses = -returns[returns < 0].sum()

    if losses == 0:
        return float("inf")

    return float(
        gains / losses
    )


def expectancy(returns):

    returns = pd.Series(returns).dropna()

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate_value = (
        len(wins) / len(returns)
        if len(returns)
        else 0
    )

    loss_rate_value = (
        len(losses) / len(returns)
        if len(returns)
        else 0
    )

    avg_win = wins.mean() if len(wins) else 0
    avg_loss = abs(losses.mean()) if len(losses) else 0

    return float(
        win_rate_value * avg_win
        -
        loss_rate_value * avg_loss
    )


def max_drawdown(returns):

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return 0.0

    equity = (
        1 + returns
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity / peak
        - 1
    )

    return float(
        drawdown.min()
    )


def backtest_summary(returns):

    returns = pd.Series(returns).dropna()

    return {
        "trades": int(len(returns)),
        "win_rate": win_rate(returns),
        "average_return": average_return(returns),
        "profit_factor": profit_factor(returns),
        "expectancy": expectancy(returns),
        "max_drawdown": max_drawdown(returns),
        "best_trade": float(
            returns.max()
        ) if len(returns) else 0.0,
        "worst_trade": float(
            returns.min()
        ) if len(returns) else 0.0,
    }


def buy_and_hold_backtest(close, horizon):

    returns = create_returns(
        close,
        horizon
    )

    return backtest_summary(
        returns
    )


def benchmark_backtest(data, horizon):

    all_returns = []

    for ticker, df in data.items():

        returns = create_returns(
            df["Close"],
            horizon
        )

        all_returns.extend(
            returns.tolist()
        )

    return backtest_summary(
        all_returns
    )