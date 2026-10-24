from core.market_data import download
from core.indicators import rsi
from core.indicators import atr
from core.ml_engine import make_model

print("Market Data OK")
print("Indicators OK")
print("ML Engine OK")

model = make_model()

print(type(model).__name__)