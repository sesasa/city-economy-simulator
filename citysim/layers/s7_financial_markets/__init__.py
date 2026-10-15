"""Stage 7: financial markets and risk sharing."""
from .risk_sharing import RiskSharing
from .stock_market import StockMarket

STAGE7_LAYERS = [StockMarket, RiskSharing]
