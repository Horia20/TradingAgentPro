import pandas as pd
import yfinance as yf


def download(ticker: str, period: str = "5y"):
    """
    Yahoo Finance downloader.
    Returnează dataframe OHLCV sau None.
    """

    try:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False
        )

    except Exception:
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    need = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in df.columns for c in need):
        return None

    df = df[need]

    df.index = pd.to_datetime(df.index)

    return df