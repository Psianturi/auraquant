from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class CorrelationSignal:
    """Output of Layer B.

    This is intentionally simple: side + confidence + evidence.
    """

    symbol: str
    lead_symbol: str
    side: Side
    confidence: float

    corr: float
    lag: int
    window: int

    why: str
    evidence_json: dict = field(default_factory=dict)
