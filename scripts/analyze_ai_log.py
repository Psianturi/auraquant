#!/usr/bin/env python3
"""Analyze AuraQuant NDJSON AI logs.

Goal
- Summarize executed trades (LONG/SHORT), realized PnL, and key stats.
- Optionally export an "offline policy" JSON derived from historical results.


Usage
  python scripts/analyze_ai_log.py
  python scripts/analyze_ai_log.py --log ai_logs/auraquant_orchestrator.ndjson
  python scripts/analyze_ai_log.py --log ai_logs/*.ndjson
  python scripts/analyze_ai_log.py --policy-out models/offline_policy.json

"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    notional_usdt: Optional[float]
    opened_at: Optional[datetime]
    closed_at: Optional[datetime]
    pnl_usdt: Optional[float]
    reason: Optional[str] = None

    @property
    def is_win(self) -> Optional[bool]:
        if self.pnl_usdt is None:
            return None
        return self.pnl_usdt > 0

    @property
    def duration_s(self) -> Optional[float]:
        if self.opened_at is None or self.closed_at is None:
            return None
        return (self.closed_at - self.opened_at).total_seconds()


def _parse_ts(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(ts))
        except Exception:
            return None
    if not isinstance(ts, str):
        return None
    s = ts.strip()
    if not s:
        return None
    # Common formats: 2026-01-20T07:48:19Z or ISO with +00:00
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _expand_logs(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        for match in glob.glob(p):
            path = Path(match)
            if path.is_file():
                out.append(path)
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def load_events(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                obj["__file"] = str(path)
                events.append(obj)

    def _sort_key(e: Dict[str, Any]) -> Tuple[int, str]:
        ts = _parse_ts(e.get("timestamp"))
        if ts is None:
            return (1, "")
        return (0, ts.isoformat())

    events.sort(key=_sort_key)
    return events


def extract_trades(events: List[Dict[str, Any]]) -> List[Trade]:
    open_by_symbol: Dict[str, List[Trade]] = {}
    closed: List[Trade] = []

    for e in events:
        stage = str(e.get("stage") or "").strip().upper()
        payload_in = e.get("input") if isinstance(e.get("input"), dict) else {}
        ts = _parse_ts(e.get("timestamp"))

        if stage == "EXECUTION":
            symbol = str(payload_in.get("symbol") or "").strip()
            side = str(payload_in.get("side") or "").strip().upper()
            if not symbol or side not in ("LONG", "SHORT"):
                continue
            t = Trade(
                symbol=symbol,
                side=side,
                entry_price=_as_float(payload_in.get("entry_price")),
                stop_loss=_as_float(payload_in.get("stop_loss")),
                take_profit=_as_float(payload_in.get("take_profit")),
                notional_usdt=_as_float(payload_in.get("notional_usdt")),
                opened_at=ts,
                closed_at=None,
                pnl_usdt=None,
            )
            open_by_symbol.setdefault(symbol, []).append(t)
            continue

        if stage == "TRADE_CLOSED":
            symbol = str(payload_in.get("symbol") or "").strip()
            pnl = _as_float(payload_in.get("pnl_usdt"))
            reason = None
            if isinstance(payload_in.get("reason"), str):
                reason = str(payload_in.get("reason"))

            if not symbol:
                continue
            stack = open_by_symbol.get(symbol) or []
            if stack:
                t = stack.pop(0)
            else:

                t = Trade(
                    symbol=symbol,
                    side="?",
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    notional_usdt=None,
                    opened_at=None,
                    closed_at=None,
                    pnl_usdt=None,
                )
            t.closed_at = ts
            t.pnl_usdt = pnl
            t.reason = reason
            closed.append(t)
            continue

    return closed


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def summarize(trades: List[Trade]) -> Dict[str, Any]:
    realized = [t for t in trades if t.pnl_usdt is not None]

    total_pnl = float(sum(t.pnl_usdt or 0.0 for t in realized))
    wins = [t for t in realized if (t.pnl_usdt or 0.0) > 0]
    losses = [t for t in realized if (t.pnl_usdt or 0.0) < 0]

    gross_profit = float(sum(t.pnl_usdt or 0.0 for t in wins))
    gross_loss = float(-sum(t.pnl_usdt or 0.0 for t in losses))

    win_rate = _safe_div(len(wins), len(realized)) if realized else 0.0
    avg_win = _safe_div(gross_profit, len(wins)) if wins else 0.0
    avg_loss = _safe_div(gross_loss, len(losses)) if losses else 0.0
    profit_factor = _safe_div(gross_profit, gross_loss) if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
    expectancy = _safe_div(total_pnl, len(realized)) if realized else 0.0

    by_symbol_side: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for t in realized:
        sym = t.symbol
        side = t.side
        bucket = by_symbol_side.setdefault(sym, {}).setdefault(side, {"n": 0, "pnl": 0.0, "wins": 0, "losses": 0})
        bucket["n"] += 1
        bucket["pnl"] += float(t.pnl_usdt or 0.0)
        if (t.pnl_usdt or 0.0) > 0:
            bucket["wins"] += 1
        elif (t.pnl_usdt or 0.0) < 0:
            bucket["losses"] += 1

    # Add derived rates
    for sym, sides in by_symbol_side.items():
        for side, b in sides.items():
            n = int(b.get("n") or 0)
            wins_n = int(b.get("wins") or 0)
            losses_n = int(b.get("losses") or 0)
            b["win_rate"] = _safe_div(wins_n, n) if n else 0.0
            b["avg_pnl"] = _safe_div(float(b.get("pnl") or 0.0), n) if n else 0.0
            b["loss_rate"] = _safe_div(losses_n, n) if n else 0.0

    return {
        "trades_total": len(trades),
        "trades_realized": len(realized),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "by_symbol_side": by_symbol_side,
    }


def build_offline_policy(summary: Dict[str, Any], min_trades: int, max_multiplier: float) -> Dict[str, Any]:
    by_symbol_side = summary.get("by_symbol_side") if isinstance(summary.get("by_symbol_side"), dict) else {}

    policy: Dict[str, Any] = {
        "type": "offline_policy_v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "min_trades": int(min_trades),
        "max_multiplier": float(max_multiplier),
        "symbols": {},
    }

    def _mult_from_edge(win_rate: float) -> float:
        edge = float(win_rate) - 0.5
        raw = 1.0 + (edge * 2.0) * float(max_multiplier)  # edge=0.1 => +0.2*max
        lo = 1.0 - float(max_multiplier)
        hi = 1.0 + float(max_multiplier)
        return float(min(max(raw, lo), hi))

    for sym, sides in by_symbol_side.items():
        if not isinstance(sides, dict):
            continue
        sym_rec: Dict[str, Any] = {}
        for side, b in sides.items():
            if side not in ("LONG", "SHORT"):
                continue
            n = int(b.get("n") or 0)
            win_rate = float(b.get("win_rate") or 0.0)
            avg_pnl = float(b.get("avg_pnl") or 0.0)
            if n < int(min_trades):
                sym_rec[side] = {
                    "n": n,
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "confidence_multiplier": 1.0,
                    "enabled": False,
                }
            else:
                sym_rec[side] = {
                    "n": n,
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "confidence_multiplier": round(_mult_from_edge(win_rate), 4),
                    "enabled": True,
                }
        if sym_rec:
            policy["symbols"][sym] = sym_rec

    return policy


def _fmt(x: float) -> str:
    if math.isinf(x):
        return "inf"
    return f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze AuraQuant AI log NDJSON trade outcomes")
    ap.add_argument(
        "--log",
        action="append",
        default=[],
        help="Path/glob to NDJSON file(s). Can be repeated. Default: ai_logs/auraquant_orchestrator.ndjson",
    )
    ap.add_argument("--json", action="store_true", help="Print summary as JSON")
    ap.add_argument("--csv-out", default="", help="Optional path to write trades CSV")
    ap.add_argument("--policy-out", default="", help="Optional path to write offline policy JSON")
    ap.add_argument("--policy-min-trades", type=int, default=20, help="Min trades per symbol/side to enable")
    ap.add_argument(
        "--policy-max-multiplier",
        type=float,
        default=0.15,
        help="Max confidence multiplier delta (0.15 => multiplier range [0.85, 1.15])",
    )

    args = ap.parse_args()

    patterns = list(args.log) if args.log else ["ai_logs/auraquant_orchestrator.ndjson"]
    paths = _expand_logs(patterns)
    if not paths:
        print("No log files found.")
        return 2

    events = load_events(paths)
    trades = extract_trades(events)
    summ = summarize(trades)

    if args.policy_out:
        policy = build_offline_policy(summ, min_trades=int(args.policy_min_trades), max_multiplier=float(args.policy_max_multiplier))
        out_path = Path(args.policy_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "symbol,side,opened_at,closed_at,duration_s,entry_price,stop_loss,take_profit,notional_usdt,pnl_usdt,reason",
        ]
        for t in trades:
            dur = t.duration_s
            lines.append(
                ",".join(
                    [
                        t.symbol,
                        t.side,
                        (t.opened_at.isoformat() if t.opened_at else ""),
                        (t.closed_at.isoformat() if t.closed_at else ""),
                        (f"{dur:.0f}" if dur is not None else ""),
                        ("" if t.entry_price is None else str(t.entry_price)),
                        ("" if t.stop_loss is None else str(t.stop_loss)),
                        ("" if t.take_profit is None else str(t.take_profit)),
                        ("" if t.notional_usdt is None else str(t.notional_usdt)),
                        ("" if t.pnl_usdt is None else str(t.pnl_usdt)),
                        ("" if not t.reason else t.reason.replace(",", " ")),
                    ]
                )
            )
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(summ, ensure_ascii=False, indent=2))
        return 0

    print(f"Files: {', '.join(str(p) for p in paths)}")
    print(f"Trades (realized): {summ['trades_realized']} / {summ['trades_total']}")
    print(
        " | ".join(
            [
                f"win_rate={_fmt(float(summ['win_rate']))}",
                f"total_pnl={_fmt(float(summ['total_pnl']))}",
                f"avg_win={_fmt(float(summ['avg_win']))}",
                f"avg_loss={_fmt(float(summ['avg_loss']))}",
                f"profit_factor={_fmt(float(summ['profit_factor']))}",
                f"expectancy={_fmt(float(summ['expectancy']))}",
            ]
        )
    )

    by = summ.get("by_symbol_side") or {}
    if isinstance(by, dict) and by:
        print("\nPer symbol/side:")
        for sym in sorted(by.keys()):
            sides = by.get(sym) or {}
            if not isinstance(sides, dict):
                continue
            for side in ("LONG", "SHORT"):
                if side not in sides:
                    continue
                b = sides[side]
                print(
                    f"- {sym} {side}: n={b.get('n')} win_rate={_fmt(float(b.get('win_rate') or 0.0))} avg_pnl={_fmt(float(b.get('avg_pnl') or 0.0))}"
                )

    if args.policy_out:
        print(f"\nWrote offline policy: {args.policy_out}")
    if args.csv_out:
        print(f"Wrote CSV: {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
