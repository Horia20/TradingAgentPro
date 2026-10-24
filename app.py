from core.market_data import download
from core.indicators import add_features

print("Loading data...")

df = download("AAPL")

if df is None:
    print("No data")
else:

    f = add_features(df)

    print("Rows:", len(f))

    print(
        f[
            [
                "Close",
                "SMA50",
                "SMA200",
                "RSI14"
            ]
        ].tail(3)
    )