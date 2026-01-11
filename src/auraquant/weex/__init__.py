"""WEEX integration utilities (MVP scaffolding).

This package includes only what we can safely build without API keys:
- constants + symbol mapping
- (later) signed private client + order manager + ai_log uploader
"""

from .symbols import (
    WEEX_ALLOWED_CONTRACT_SYMBOLS,
    to_weex_contract_symbol,
    is_allowed_contract_symbol,
)

__all__ = [
    "WEEX_ALLOWED_CONTRACT_SYMBOLS",
    "to_weex_contract_symbol",
    "is_allowed_contract_symbol",
]
