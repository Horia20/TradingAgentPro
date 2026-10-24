from core.market_data import download
from core.ml_engine import train_model

data = {
    "AAPL": download("AAPL"),
    "MSFT": download("MSFT"),
}

result = train_model(
    data=data,
    horizon=10,
    target=0.05
)

print(result)