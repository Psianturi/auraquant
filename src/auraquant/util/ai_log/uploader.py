from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests


def iter_ndjson(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def chunked(items: Iterable[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass
class AiLogBatchUploader:

    url: str
    timeout_seconds: float = 15.0
    headers: Optional[Dict[str, str]] = None

    def upload_batch(self, events: List[Dict[str, Any]]) -> requests.Response:
        hdrs = {"Content-Type": "application/json"}
        if self.headers:
            hdrs.update(self.headers)
        resp = requests.post(self.url, json=events, headers=hdrs, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp


def _parse_headers(header_kv: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kv in header_kv:
        if ":" not in kv:
            raise ValueError(f"Invalid header '{kv}'. Use 'Key:Value'.")
        k, v = kv.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Upload AuraQuant AI log NDJSON in batches.")
    ap.add_argument("--file", required=True, help="Path to NDJSON log file (e.g., ai_logs/demo_ai_log.ndjson)")
    ap.add_argument("--url", default=os.environ.get("AURAQUANT_AI_LOG_UPLOAD_URL"), help="Upload endpoint URL")
    ap.add_argument("--batch-size", type=int, default=200, help="Number of events per POST")
    ap.add_argument("--header", action="append", default=[], help="Extra HTTP header, e.g. 'Authorization: Bearer XXX'")
    ap.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds")
    args = ap.parse_args(argv)

    if not args.url:
        ap.error("--url is required (or set AURAQUANT_AI_LOG_UPLOAD_URL)")

    headers = _parse_headers(args.header)
    uploader = AiLogBatchUploader(url=str(args.url), timeout_seconds=float(args.timeout), headers=headers)

    total = 0
    for batch in chunked(iter_ndjson(args.file), batch_size=max(int(args.batch_size), 1)):
        uploader.upload_batch(batch)
        total += len(batch)

    print(f"Uploaded {total} AI log events to {args.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
