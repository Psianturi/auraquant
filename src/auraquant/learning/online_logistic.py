from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .features import FeatureVector


def _sigmoid(z: float) -> float:
    z = float(z)
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class OnlineLogisticModel:
    """Tiny online logistic regression (SGD) with L2 regularization.

    Purpose: learn P(win) from trade features.
    No external deps; transparent weights; safe to use as a *gating/weighting* signal.
    """

    lr: float = 0.25
    l2: float = 1e-3
    bias: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    seen: int = 0

    def predict_proba(self, x: FeatureVector) -> float:
        z = float(self.bias)
        for k, v in x.to_dict().items():
            z += float(self.weights.get(k, 0.0)) * float(v)
        return float(_sigmoid(z))

    def update(self, x: FeatureVector, y: int) -> float:
        """Single SGD update. Returns predicted proba BEFORE update."""

        y = 1 if int(y) != 0 else 0
        p = self.predict_proba(x)
        err = (p - float(y))

        self.bias -= self.lr * err

        # Weights
        for k, v in x.to_dict().items():
            w = float(self.weights.get(k, 0.0))
            grad = err * float(v) + self.l2 * w
            w -= self.lr * grad
            self.weights[k] = float(w)

        self.seen += 1
        return float(p)

    def to_json(self) -> Dict[str, object]:
        return {
            "type": "online_logistic_v1",
            "lr": self.lr,
            "l2": self.l2,
            "bias": self.bias,
            "weights": dict(self.weights),
            "seen": self.seen,
        }

    @classmethod
    def from_json(cls, obj: Dict[str, object]) -> "OnlineLogisticModel":
        if obj.get("type") != "online_logistic_v1":
            raise ValueError("Unsupported model type")
        m = cls(
            lr=float(obj.get("lr", 0.25)),
            l2=float(obj.get("l2", 1e-3)),
            bias=float(obj.get("bias", 0.0)),
        )
        weights = obj.get("weights", {})
        if isinstance(weights, dict):
            m.weights = {str(k): float(v) for k, v in weights.items()}
        m.seen = int(obj.get("seen", 0))
        return m

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "OnlineLogisticModel":
        p = Path(path)
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("Invalid model JSON")
        return cls.from_json(obj)
