from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AiLogEvent:
    """Local AI log event shaped similarly to WEEX uploadAiLog payload.

    WEEX uploadAiLog required fields:
    - stage (str)
    - model (str)
    - input (json)
    - output (json)
    - explanation (str <= 1000 chars)
    - orderId (optional)

    We store it locally as NDJSON so it can be uploaded later once API keys exist.
    """

    stage: str
    model: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    explanation: str
    order_id: Optional[int] = None
    timestamp: Optional[datetime] = None

    def to_payload(self) -> Dict[str, Any]:
        exp = (self.explanation or "").strip()
        if len(exp) > 1000:
            exp = exp[:1000]
        return {
            "orderId": self.order_id,
            "stage": self.stage,
            "model": self.model,
            "input": self.input,
            "output": self.output,
            "explanation": exp,
        }


class AiLogStore:
    """Append-only NDJSON store for AI logs.

    File example: ai_logs/ai_log.ndjson
    """

    def __init__(self, path: str | Path = "ai_logs/ai_log.ndjson"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AiLogEvent) -> None:
        payload = event.to_payload()
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
