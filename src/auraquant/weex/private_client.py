from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


def _json_dumps_compact(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


@dataclass
class WeexPrivateRestClient:

    base_url: str = "https://api-contract.weex.com"
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    passphrase: Optional[str] = None
    locale: str = "en-US"
    timeout_seconds: int = 15

    def __post_init__(self) -> None:
        self.base_url = os.getenv("WEEX_BASE_URL", self.base_url).rstrip("/")
        if self.api_key is None:
            self.api_key = os.getenv("WEEX_API_KEY")
        if self.secret_key is None:
            self.secret_key = os.getenv("WEEX_SECRET_KEY")
        if self.passphrase is None:
            self.passphrase = os.getenv("WEEX_PASSPHRASE")
        self._session = requests.Session()

    def require_env(self) -> None:
        missing: list[str] = []
        if not self.api_key:
            missing.append("WEEX_API_KEY")
        if not self.secret_key:
            missing.append("WEEX_SECRET_KEY")
        if not self.passphrase:
            missing.append("WEEX_PASSPHRASE")
        if missing:
            raise RuntimeError("Missing required env vars: " + ", ".join(missing))

    def _timestamp_ms(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, timestamp_ms: str, method: str, path: str, query: str = "", body: str = "") -> str:
        assert self.secret_key is not None
        message = timestamp_ms + method.upper() + path + query + body
        sig = hmac.new(self.secret_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).digest()
        return base64.b64encode(sig).decode("utf-8")

    def _headers(self, timestamp_ms: str, signature: str) -> Dict[str, str]:
        assert self.api_key is not None
        assert self.passphrase is not None
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp_ms,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": self.locale,
        }

    def signed_get(self, path: str, query: str = "") -> requests.Response:
        self.require_env()
        ts = self._timestamp_ms()
        sig = self._sign(ts, "GET", path, query)
        headers = self._headers(ts, sig)
        url = f"{self.base_url}{path}{query}"
        return self._session.get(url, headers=headers, timeout=self.timeout_seconds)

    def signed_post(self, path: str, body_obj: Dict[str, Any], query: str = "") -> requests.Response:
        self.require_env()
        body = _json_dumps_compact(body_obj)
        ts = self._timestamp_ms()
        sig = self._sign(ts, "POST", path, query, body)
        headers = self._headers(ts, sig)
        url = f"{self.base_url}{path}{query}"
        return self._session.post(url, headers=headers, data=body, timeout=self.timeout_seconds)
