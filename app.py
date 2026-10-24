from core.market_data import download
from core.indicators import rsi
from core.indicators import atr
from core.ml_engine import make_model
from core.dataset_engine import non_overlapping

print("Market Data OK")
print("Indicators OK")
print("ML Engine OK")
print("Dataset Engine OK")

model = make_model()

print(type(model).__name__)