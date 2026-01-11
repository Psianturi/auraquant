# AuraQuant

**A Hybrid Multi-Agent AI Trading System for WEEX AI Hackathon**

AuraQuant is an AI-driven trading agent integrating NLP sentiment analysis, cross-asset correlation modeling, and dynamic risk management.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│         (State Machine: SCAN→QUALIFY→ENTER→MANAGE→          │
│                    EXIT→RECONCILE)                          │
└──────────────┬──────────────┬──────────────┬───────────────┘
               │              │              │
    ┌──────────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
    │   LAYER A       │ │  LAYER B  │ │   LAYER C   │
    │   Sentiment     │ │Correlation│ │ RiskEngine  │
    │   (The Brain)   │ │(Trigger)  │ │(Guardian)   │
    └─────────────────┘ └───────────┘ └─────────────┘
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                    ┌─────────────────┐
                    │    EXECUTION    │
                    │  (Paper / WEEX) │
                    └─────────────────┘
```

### Three Pillars

| Layer | Role | Output |
|-------|------|--------|
| **A - Sentiment** | Analyze news with half-life decay | `LONG / SHORT / NEUTRAL` |
| **B - Correlation** | Confirm entry via BTC lead-lag | `CorrelationSignal` |
| **C - Risk** | Final gate: sizing, SL/TP, circuit breaker | `APPROVED / DENIED` |

## Key Features

- **State Machine** - Prevents order spam and race conditions
- **Circuit Breaker** - Auto-stop on daily drawdown (-2%) or loss streak (3)
- **ATR-based SL/TP** - Dynamic stops based on volatility
- **Online Learning** - Adaptive P(win) estimation from trade outcomes
- **AI Log Store** - NDJSON format ready for WEEX upload

## Quick Start

```bash
pip install -r requirements.txt

# Demo (paper trading with synthetic data)
python src/demo_orchestrator.py

# WEEX public API test
python src/weex_public_smoketest.py
```

## Project Structure

```
src/auraquant/
├── core/           # Orchestrator, state machine
├── sentiment/      # SentimentProcessor, NewsProvider
├── correlation/    # CorrelationTrigger (BTC lead-lag)
├── risk/           # RiskEngine, CircuitBreaker
├── execution/      # PaperOrderManager
├── learning/       # OnlineLogistic P(win) model
├── data/           # Price providers (Static, WEEX REST)
├── weex/           # Symbol allowlist
└── util/           # JSON logging, AI log store
```


## Configuration

**OrchestratorConfig:**
- `symbol = "SOL/USDT"` - Target pair
- `default_leverage = 10.0` - Conservative
- `min_confidence = 0.3` - Trade threshold

**RiskEngine:**
- `daily_drawdown_limit_pct = -2.0`
- `max_consecutive_losses = 3`
- `risk_per_trade_pct = 0.5`
- `sl_atr_mult = 1.5`, `tp_atr_mult = 3.0`

## AI Evidence Logging

```json
{
  "module": "RiskEngine",
  "decision": "APPROVED",
  "metrics": {
    "equity_now": 1000.0,
    "stop_loss": 97.73,
    "take_profit": 99.98
  }
}
```


