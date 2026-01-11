from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


def utc_iso(ts: datetime | None = None) -> str:
    ts = ts or datetime.utcnow()
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    ts = ts.replace(microsecond=0)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def log_json(logger: logging.Logger, payload: Dict[str, Any], level: int = logging.INFO) -> None:
    """Log JSON in a consistent, judge-friendly way."""

    logger.log(level, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
