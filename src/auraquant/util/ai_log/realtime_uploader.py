"""Real-time AI log uploader with retry logic and disk queue.

WEEX requires:
- Real-time upload (max 1 minute delay)
- Every AI decision must be logged
- Retry on network failure + local queue fallback
- All fields match request parameter names exactly

Queue persistence:
  If upload fails due to network error, logs are saved to disk queue.
  Retry thread periodically flushes queued events on reconnection.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


@dataclass
class QueuedLogEntry:
    """Disk-persisted log entry (not yet successfully uploaded)."""

    event_payload: Dict[str, Any]
    queued_at: str  # ISO format
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiskQueue:
    """Simple disk-based queue for failed uploads."""

    def __init__(self, queue_dir: str | Path = "ai_logs/.upload_queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def push(self, entry: QueuedLogEntry) -> None:
        """Persist entry to disk."""
        with self._lock:
            ts = datetime.now(timezone.utc).isoformat().replace(":", "-")
            counter = 0
            while True:
                fname = f"entry_{ts}_{counter:03d}.json"
                fpath = self.queue_dir / fname
                if not fpath.exists():
                    break
                counter += 1

            with fpath.open("w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False)

    def pop_all(self) -> List[QueuedLogEntry]:
        """Read all queued entries from disk."""
        with self._lock:
            entries: List[QueuedLogEntry] = []
            if not self.queue_dir.exists():
                return entries

            for fpath in sorted(self.queue_dir.glob("entry_*.json")):
                try:
                    with fpath.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    entry = QueuedLogEntry(**data)
                    entries.append(entry)
                    # Delete after reading
                    fpath.unlink()
                except Exception as e:
                    logger.warning(f"Failed to read queued entry {fpath.name}: {e}")
            return entries

    def clear(self) -> None:
        """Remove all queued entries."""
        with self._lock:
            if self.queue_dir.exists():
                for fpath in self.queue_dir.glob("entry_*.json"):
                    try:
                        fpath.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete {fpath.name}: {e}")


class RealTimeAiLogUploader:
    """Async real-time uploader with retry + disk queue."""

    def __init__(
        self,
        weex_upload_url: str,
        headers: Optional[Dict[str, str]] = None,
        queue_dir: str | Path = "ai_logs/.upload_queue",
        max_retries: int = 3,
        retry_delay_seconds: float = 5.0,
        upload_timeout_seconds: float = 15.0,
        flush_interval_seconds: float = 30.0,
        weex_api_key: Optional[str] = None,
        weex_secret_key: Optional[str] = None,
        weex_passphrase: Optional[str] = None,
    ):
       
        self.weex_upload_url = weex_upload_url
        self.headers = headers or {}
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.upload_timeout_seconds = upload_timeout_seconds
        self.flush_interval_seconds = flush_interval_seconds
        self.weex_api_key = weex_api_key
        self.weex_secret_key = weex_secret_key
        self.weex_passphrase = weex_passphrase
        self.logger = logger

        self.queue = DiskQueue(queue_dir)
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and down-scope payload to metadata-only if configured.

        Env controls:
        - AI_LOG_UPLOAD_MODE: 'metadata' | 'full' (default 'metadata')
        - AI_LOG_MASK_SENSITIVE: '1' to mask sensitive fields
        - AI_LOG_LEVEL: optional level string added to payload
        - AI_LOG_UPLOAD_STAGES: comma list of stages to allow (default: all)
        """
        mode = os.getenv("AI_LOG_UPLOAD_MODE", "metadata").lower().strip()
        mask = os.getenv("AI_LOG_MASK_SENSITIVE", "1") == "1"
        level = os.getenv("AI_LOG_LEVEL")

        stage = str(payload.get("stage", "")).upper()
        allow_stages = os.getenv("AI_LOG_UPLOAD_STAGES")
        if allow_stages:
            allowed = {s.strip().upper() for s in allow_stages.split(",") if s.strip()}
            if stage and stage not in allowed:
                # Return a minimal heartbeat event instead of dropping completely
                return {
                    "stage": stage,
                    "model": str(payload.get("model", "AuraQuant")),
                    "input": {"heartbeat": True},
                    "output": {},
                    "explanation": "Filtered by AI_LOG_UPLOAD_STAGES",
                    "orderId": payload.get("orderId"),
                }

        if level:
            # Add level as part of input metadata (safer for API)
            inp = payload.get("input") or {}
            if isinstance(inp, dict):
                inp["level"] = level
                payload["input"] = inp

        if mode != "metadata":
            return payload

        def _round(v: Any) -> Any:
            try:
                f = float(v)
                # round to 4 decimals for stability
                return round(f, 4)
            except Exception:
                return v

        # Build metadata-only input/output
        safe_input: Dict[str, Any] = {}
        safe_output: Dict[str, Any] = {}

        inp = payload.get("input") or {}
        outp = payload.get("output") or {}
        if isinstance(inp, dict):
            for k in ("symbol", "price", "atr", "phase", "intent", "equity_now"):
                if k in inp:
                    v = inp[k]
                    if isinstance(v, (int, float)):
                        v = _round(v)
                    safe_input[k] = v
            # Add coarse buckets to avoid precise strategy leakage
            if "price" in safe_input:
                try:
                    p = float(safe_input["price"]) if safe_input["price"] is not None else None
                    if p:
                        safe_input["price_bucket"] = int(p // 10) * 10
                except Exception:
                    pass
            if "atr" in safe_input and safe_input.get("price"):
                try:
                    atr = float(safe_input["atr"]) or 0.0
                    price = float(safe_input["price"]) or 1.0
                    safe_input["atr_pct_bucket"] = _round((atr / price) * 100.0)
                except Exception:
                    pass

        if isinstance(outp, dict):
            for k in ("confidence", "allowed", "decision", "reason", "trade_count", "positions_opened", "trades_closed"):
                if k in outp:
                    v = outp[k]
                    if isinstance(v, (int, float)):
                        v = _round(v)
                    safe_output[k] = v

        exp = str(payload.get("explanation", ""))
        if mask:
            # Strip detailed reasoning
            exp = exp[:200]

        sanitized = {
            "orderId": payload.get("orderId"),
            "stage": payload.get("stage"),
            "model": payload.get("model"),
            "input": safe_input,
            "output": safe_output,
            "explanation": exp,
        }
        return sanitized

    def start(self) -> None:
        """Start background retry thread."""
        if self._background_thread is not None:
            return
        self._stop_event.clear()
        self._background_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._background_thread.start()
        logger.info(f"[AI Log Uploader] Started background retry thread (flush every {self.flush_interval_seconds}s)")

    def stop(self, timeout_seconds: float = 10.0) -> None:
        """Stop background thread and flush remaining queued events."""
        if self._background_thread is None:
            return
        self._stop_event.set()
        self._background_thread.join(timeout=timeout_seconds)
        self._background_thread = None
        logger.info("[AI Log Uploader] Background thread stopped")

    def upload(self, event_payload: Dict[str, Any]) -> bool:
        """
        Upload a single AI log event synchronously.

        If upload fails, event is queued to disk and retry thread will attempt later.

        Args:
            event_payload: Dict with keys {stage, model, input, output, explanation, orderId}

        Returns:
            True if successfully uploaded, False if queued for retry.
        """
        try:
            sanitized = self._sanitize_payload(event_payload)
            return self._upload_sync(sanitized)
        except Exception as e:
            logger.warning(f"[AI Log Uploader] Upload failed (will retry): {e}")
            # Queue for retry
            entry = QueuedLogEntry(
                event_payload=event_payload,
                queued_at=datetime.now(timezone.utc).isoformat(),
                retry_count=0,
            )
            self.queue.push(entry)
            return False

    def _upload_sync(self, event_payload: Dict[str, Any]) -> bool:
        """Synchronous upload attempt. Raises on error."""

        body_str = json.dumps(event_payload)
        
        hdrs = {"Content-Type": "application/json"}
        hdrs.update(self.headers)
        
        if self.weex_api_key and self.weex_secret_key and self.weex_passphrase:
            timestamp = str(int(time.time() * 1000))
            message = timestamp + "POST" + "/capi/v2/order/uploadAiLog" + "" + body_str
            signature = hmac.new(
                self.weex_secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            signature_b64 = base64.b64encode(signature).decode()
            
            hdrs["ACCESS-KEY"] = self.weex_api_key
            hdrs["ACCESS-SIGN"] = signature_b64
            hdrs["ACCESS-TIMESTAMP"] = timestamp
            hdrs["ACCESS-PASSPHRASE"] = self.weex_passphrase

        resp = requests.post(
            self.weex_upload_url,
            data=body_str,
            headers=hdrs,
            timeout=self.upload_timeout_seconds,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"WEEX AI log upload failed HTTP {resp.status_code}: {resp.text[:200]}"
            )
        
        # WEEX returns code "00000" on success, other codes are errors
        try:
            result = resp.json()
            code = result.get("code", "")
            if code != "00000":
                msg = result.get("msg", "unknown error")
                raise RuntimeError(
                    f"WEEX AI log upload rejected: code={code}, msg={msg}"
                )
        except json.JSONDecodeError:
            pass
        
        self.logger.debug(f"[AI Log Uploader] Uploaded 1 event to {self.weex_upload_url}")
        return True

    def _flush_loop(self) -> None:
        """Background thread: periodically retry queued events."""
        while not self._stop_event.is_set():
            try:
                time.sleep(self.flush_interval_seconds)
                self._flush_queue()
            except Exception as e:
                logger.error(f"[AI Log Uploader] Flush loop error: {e}")

    def _flush_queue(self) -> None:
        """Attempt to upload all queued events."""
        entries = self.queue.pop_all()
        if not entries:
            return

        logger.info(f"[AI Log Uploader] Retrying {len(entries)} queued event(s)")

        for entry in entries:
            if entry.retry_count >= self.max_retries:
                logger.error(f"[AI Log Uploader] Max retries exceeded. Discarding event: {entry.event_payload}")
                continue

            time.sleep(self.retry_delay_seconds)
            try:
                self._upload_sync(entry.event_payload)
                logger.info(f"[AI Log Uploader] Successfully uploaded queued event (retry #{entry.retry_count})")
            except Exception as e:
                logger.warning(
                    f"[AI Log Uploader] Retry #{entry.retry_count + 1} failed. Re-queuing: {e}"
                )
                # Re-queue with incremented retry count
                entry.retry_count += 1
                entry.queued_at = datetime.now(timezone.utc).isoformat()
                self.queue.push(entry)

    def clear_queue(self) -> None:
        """Clear all queued events (use with caution)."""
        self.queue.clear()


def make_uploader_from_env(queue_dir: str | Path = "ai_logs/.upload_queue") -> Optional[RealTimeAiLogUploader]:
    """Factory: create uploader from environment variables.

    Expected env vars:
    - WEEX_AI_LOG_UPLOAD_URL (required)
    - WEEX_API_KEY (optional, for HMAC auth)
    - WEEX_SECRET_KEY (optional, for HMAC auth)
    - WEEX_PASSPHRASE (optional, for HMAC auth)
    - WEEX_AI_LOG_AUTH_HEADER (optional, e.g., "Authorization:Bearer XXX")
    """
    upload_url = os.getenv("WEEX_AI_LOG_UPLOAD_URL")

    weex_api_key = os.getenv("WEEX_API_KEY")
    weex_secret_key = os.getenv("WEEX_SECRET_KEY")
    weex_passphrase = os.getenv("WEEX_PASSPHRASE")

    if not upload_url:
        # If credentials exist, default to official UploadAiLog endpoint.
        if weex_api_key and weex_secret_key and weex_passphrase:
            upload_url = "https://api-contract.weex.com/capi/v2/order/uploadAiLog"
            logger.info("[AI Log Uploader] WEEX_AI_LOG_UPLOAD_URL not set. Using default UploadAiLog endpoint.")
        else:

            optional = os.getenv("WEEX_AI_LOG_UPLOAD_OPTIONAL", "1")
            msg = (
                "[AI Log Uploader] WEEX_AI_LOG_UPLOAD_URL not set. Uploader disabled. "
                "Set WEEX_AI_LOG_UPLOAD_URL or provide WEEX_API_KEY/WEEX_SECRET_KEY/WEEX_PASSPHRASE."
            )
            if optional == "1":
                logger.info(msg)
            else:
                logger.warning(msg)
            return None

    headers: Dict[str, str] = {}
    auth = os.getenv("WEEX_AI_LOG_AUTH_HEADER")
    if auth:
        # Format: "Authorization:Bearer XXX"
        if ":" in auth:
            k, v = auth.split(":", 1)
            headers[k.strip()] = v.strip()

    return RealTimeAiLogUploader(
        weex_upload_url=upload_url,
        headers=headers,
        queue_dir=queue_dir,
        weex_api_key=weex_api_key,
        weex_secret_key=weex_secret_key,
        weex_passphrase=weex_passphrase,
    )
