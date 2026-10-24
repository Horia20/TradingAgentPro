import pandas as pd

from core.market_data import download
from core.ml_engine import train_model, predict_stock


TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "META",
    "GOOGL",
    "AMZN",
    "TSLA",
]

data = {}

for ticker in TICKERS:

    print(f"Loading {ticker}")

    df = download(ticker)

    if df is not None:
        data[ticker] = df

result = train_model(
    data=data,
    horizon=10,
    target=0.05,
)

if result is None:
    print("Model training failed.")
    raise SystemExit

print("\n=== MODEL RESULTS ===\n")

print(
    f"AUC: {result['metrics']['auc']:.3f}"
)

print(
    f"Train rows: {result['metrics']['n_train']}"
)

print(
    f"Test rows: {result['metrics']['n_test']}"
)

print("\n=== STOCK SCAN ===\n")

model = result["model"]

scores = []

for ticker, df in data.items():

    prob = predict_stock(
        model,
        df
    )

    scores.append(
        (
            ticker,
            prob
        )
    )

scores.sort(
    key=lambda x: x[1],
    reverse=True
)

results = []

for ticker, prob in scores:

    print(
        f"{ticker:8} {prob:.2%}"
    )

    rating = (
        "BUY"
        if prob >= 0.40
        else "WATCH"
        if prob >= 0.20
        else "AVOID"
    )

    results.append(
        {
            "ticker": ticker,
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100, 2),
            "rating": rating,
        }
    )

df_results = pd.DataFrame(results)

df_results.to_csv(
    "reports/scan_results.csv",
    index=False
)

print("\n=== TOP PICKS ===\n")

print(df_results)

print("\nSaved: reports/scan_results.csv")
