"""Market data providers for AuraQuant (MVP)."""

from .price_provider import PriceProvider, StaticPriceProvider
from .multi_price_provider import MultiPriceProvider, StaticMultiPriceProvider

try:
	from .coingecko_client import CoinGeckoClient, pick_weex_contract_symbol_by_liquidity
except ModuleNotFoundError:
	CoinGeckoClient = None
	pick_weex_contract_symbol_by_liquidity = None

# Optional: requires 'requests'
try:
	from .weex_rest_price_provider import WeexRestMultiPriceProvider
except ModuleNotFoundError: 
	WeexRestMultiPriceProvider = None

__all__ = [
	"PriceProvider",
	"StaticPriceProvider",
	"MultiPriceProvider",
	"StaticMultiPriceProvider",
	"WeexRestMultiPriceProvider",
	"CoinGeckoClient",
	"pick_weex_contract_symbol_by_liquidity",
]
