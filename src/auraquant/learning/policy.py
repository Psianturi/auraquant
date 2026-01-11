from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .features import FeatureVector
from .online_logistic import OnlineLogisticModel


@dataclass
class TradePolicyLearner:
    """Thin wrapper that persists an OnlineLogisticModel to disk.

    The orchestrator uses it to:
    - score a candidate intent (P(win))
    - update after trade close (win/loss)

    This is intentionally simple and auditable.
    """

    model_path: str = "models/trade_policy.json"
    model: OnlineLogisticModel = None 

    def __post_init__(self) -> None:
        try:
            self.model = OnlineLogisticModel.load(self.model_path)
        except Exception:
            self.model = OnlineLogisticModel()

    def score(self, x: FeatureVector) -> float:
        return float(self.model.predict_proba(x))

    def update(self, x: FeatureVector, is_win: bool) -> Tuple[float, int]:
        p_before = float(self.model.update(x, 1 if is_win else 0))
        self.model.save(self.model_path)
        return p_before, int(self.model.seen)

    def weights(self) -> dict:
        return dict(self.model.weights)

    def seen(self) -> int:
        return int(self.model.seen)
