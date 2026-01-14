from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
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

    def _timestamp_iso(self) -> Optional[str]:
        if self.timestamp is None:
            return None
        ts = self.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat().replace("+00:00", "Z")

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

    def to_record(self) -> Dict[str, Any]:
        """Local NDJSON record for audit/debug.

        Includes a timestamp when available, but keeps WEEX payload shape intact.
        The uploader strips any extra keys before uploading.
        """

        rec = dict(self.to_payload())
        ts = self._timestamp_iso()
        if ts:
            rec["timestamp"] = ts
        return rec


class AiLogStore:
    """Append-only NDJSON store for AI logs.

    File example: ai_logs/ai_log.ndjson
    """

    def __init__(self, path: str | Path = "ai_logs/ai_log.ndjson"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AiLogEvent) -> None:
        record = event.to_record()
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
