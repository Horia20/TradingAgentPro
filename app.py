import pandas as pd

from core.market_data import download
from core.ml_engine import (
    train_model,
    predict_stock,
    build_dataset,
)

from core.scoring_engine import ai_score

from core.backtest_engine import (
    ai_strategy_summary,
)

from core.walkforward_engine import (
    walk_forward_validation,
)


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
    f"Precision: {result['metrics']['precision']:.3f}"
)

print(
    f"Recall: {result['metrics']['recall']:.3f}"
)

print(
    f"F1: {result['metrics']['f1']:.3f}"
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

rank = 1

for ticker, prob in scores:

    score, rating = ai_score(prob)

    print(
        f"{ticker:8} {prob:.2%}"
    )

    results.append(
        {
            "rank": rank,
            "ticker": ticker,
            "score": score,
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100, 2),
            "rating": rating,
        }
    )

    rank += 1

df_results = pd.DataFrame(results)

df_results.to_csv(
    "reports/scan_results.csv",
    index=False
)

print("\n=== TOP PICKS ===\n")

print(df_results)

print("\nSaved: reports/scan_results.csv")


print("\n=== AI STRATEGY TEST ===\n")

test_set = result["test_set"]

print("\nThreshold 0.30")
print(
    ai_strategy_summary(
        test_set,
        threshold=0.30
    )
)

print("\nThreshold 0.40")
print(
    ai_strategy_summary(
        test_set,
        threshold=0.40
    )
)

print("\nThreshold 0.50")
print(
    ai_strategy_summary(
        test_set,
        threshold=0.50
    )
)

print("\nThreshold 0.60")
print(
    ai_strategy_summary(
        test_set,
        threshold=0.60
    )
)


print("\n=== WALK FORWARD TEST ===\n")

dataset = build_dataset(
    data=data,
    horizon=10,
    target=0.05
)

wf = walk_forward_validation(
    dataset,
    folds=4,
    threshold=0.40
)

print(wf)

print("\n=== WALK FORWARD AVERAGES ===\n")

print(
    wf.mean(
        numeric_only=True
    )
)