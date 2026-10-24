from core.market_data import download
from core.ml_engine import build_dataset

data = {
    "AAPL": download("AAPL"),
    "MSFT": download("MSFT")
}

ds = build_dataset(
    data=data,
    horizon=10,
    target=0.05
)

print("Dataset shape:")
print(ds.shape)

print()
print(ds.head())