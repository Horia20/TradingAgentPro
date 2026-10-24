import pandas as pd


def forward_return(close, horizon):

    return (
        close.shift(-horizon)
        / close
        - 1
    )