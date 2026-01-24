from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class TradeStatsStore:
    """Persist simple per-symbol win/loss counts
    """

    path: str = os.getenv("TRADE_STATS_PATH", "models/trade_stats.json")
    wins: Dict[str, int] = field(default_factory=dict)
    losses: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            p = Path(self.path)
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    self.wins = {str(k): int(v) for k, v in obj.get("wins", {}).items()}
                    self.losses = {str(k): int(v) for k, v in obj.get("losses", {}).items()}
        except Exception:
            # Start fresh if load fails
            self.wins = {}
            self.losses = {}

    def record(self, symbol: str, is_win: bool) -> None:
        symbol = str(symbol)
        if is_win:
            self.wins[symbol] = int(self.wins.get(symbol, 0)) + 1
        else:
            self.losses[symbol] = int(self.losses.get(symbol, 0)) + 1

    def win_rate(self, symbol: str) -> float:
        w = int(self.wins.get(symbol, 0))
        l = int(self.losses.get(symbol, 0))
        # Laplace smoothing
        return float((w + 1) / (w + l + 2))

    def save(self) -> None:
        try:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            obj = {"wins": self.wins, "losses": self.losses, "type": "trade_stats_v1"}
            p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return
