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
            return self._upload_sync(event_payload)
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

        payload = [event_payload]
        body_str = json.dumps(payload)
        
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
