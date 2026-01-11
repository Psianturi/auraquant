"""Market data providers for AuraQuant (MVP)."""

from .price_provider import PriceProvider, StaticPriceProvider
from .multi_price_provider import MultiPriceProvider, StaticMultiPriceProvider

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
]
