#!/usr/bin/env python3
"""
Parse bot log files to extract key trading events for learning/analysis.
Extracts: SCAN, QUALIFY, ENTER, POSITION_OPENED, TRADE_CLOSED, denials, etc.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict


def parse_log_file(log_path: Path) -> Dict[str, Any]:
    events = []
    stats = defaultdict(int)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: YYYY-MM-DD HH:MM:SS,mmm [LEVEL] module: {json}
            json_match = re.search(r'\{.*\}', line)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    phase = data.get('phase')
                    event = data.get('event')
                    module = data.get('module')
                    
                    if phase:
                        stats[f'phase_{phase}'] += 1
                    if event:
                        stats[f'event_{event}'] += 1
                    
                    if phase == 'SCAN':
                        events.append({
                            'type': 'SCAN',
                            'timestamp': data.get('timestamp'),
                            'symbol': data.get('symbol'),
                            'price': data.get('price'),
                            'sentiment': data.get('sentiment', {})
                        })
                    
                    elif phase == 'QUALIFY':
                        deny_reason = data.get('deny')
                        if deny_reason:
                            stats[f'deny_{deny_reason}'] += 1
                            events.append({
                                'type': 'QUALIFY_DENIED',
                                'timestamp': data.get('timestamp'),
                                'symbol': data.get('symbol'),
                                'reason': deny_reason,
                                'confidence': data.get('confidence'),
                                'min_confidence': data.get('min_confidence')
                            })
                        else:
                            stats['qualify_approved'] += 1
                            events.append({
                                'type': 'QUALIFY_APPROVED',
                                'timestamp': data.get('timestamp'),
                                'symbol': data.get('symbol'),
                                'intent': data.get('intent'),
                                'p_win': data.get('p_win')
                            })
                    
                    elif phase == 'ENTER':
                        events.append({
                            'type': 'ENTER',
                            'timestamp': data.get('timestamp'),
                            'symbol': data.get('symbol'),
                            'risk_decision': data.get('risk_decision')
                        })
                    
                    elif event == 'POSITION_OPENED':
                        stats['positions_opened'] += 1
                        events.append({
                            'type': 'POSITION_OPENED',
                            'timestamp': data.get('timestamp'),
                            'symbol': data.get('symbol'),
                            'side': data.get('side'),
                            'entry_price': data.get('entry_price'),
                            'order_id': data.get('order_id'),
                            'stop_loss': data.get('stop_loss'),
                            'take_profit': data.get('take_profit'),
                            'notional_usdt': data.get('notional_usdt')
                        })
                    
                    elif event == 'TRADE_CLOSED' or module == 'TradeResult':
                        stats['trades_closed'] += 1
                        pnl = data.get('pnl_usdt') or data.get('pnl')
                        if pnl and float(pnl) > 0:
                            stats['trades_won'] += 1
                        elif pnl and float(pnl) < 0:
                            stats['trades_lost'] += 1
                        
                        events.append({
                            'type': 'TRADE_CLOSED',
                            'timestamp': data.get('timestamp'),
                            'symbol': data.get('symbol'),
                            'pnl_usdt': pnl,
                            'reason': data.get('reason') or data.get('trigger'),
                            'held_for_seconds': data.get('held_for_seconds')
                        })
                    
                    elif module == 'RiskEngine':
                        decision = data.get('decision')
                        if decision == 'DENIED':
                            reason = data.get('reason', 'UNKNOWN')
                            stats[f'risk_deny_{reason}'] += 1
                
                except json.JSONDecodeError:
                    continue
            
            if '[WARNING]' in line or '[ERROR]' in line:
                if 'CONFIDENCE_TOO_LOW' in line:
                    stats['warn_confidence_low'] += 1
                elif 'ATR' in line and 'too low' in line.lower():
                    stats['warn_atr_low'] += 1
                elif 'SIDEWAYS' in line:
                    stats['warn_sideways'] += 1
    
    return {
        'file': log_path.name,
        'events': events,
        'stats': dict(stats)
    }


def parse_ndjson_file(ndjson_path: Path) -> Dict[str, Any]:
    """Parse a .ndjson AI log file."""
    events = []
    stats = defaultdict(int)
    
    with open(ndjson_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                stage = data.get('stage')
                
                if stage:
                    stats[f'stage_{stage}'] += 1
                
                if stage == 'SCAN':
                    events.append({
                        'type': 'AI_SCAN',
                        'timestamp': data.get('timestamp'),
                        'model': data.get('model'),
                        'input': data.get('input', {}),
                        'output': data.get('output', {})
                    })
                
                elif stage == 'QUALIFY':
                    events.append({
                        'type': 'AI_QUALIFY',
                        'timestamp': data.get('timestamp'),
                        'explanation': data.get('explanation')
                    })
                
                elif stage == 'ENTER':
                    events.append({
                        'type': 'AI_ENTER',
                        'timestamp': data.get('timestamp'),
                        'order_id': data.get('order_id'),
                        'input': data.get('input', {})
                    })
            
            except json.JSONDecodeError:
                continue
    
    return {
        'file': ndjson_path.name,
        'events': events,
        'stats': dict(stats)
    }


def main():
    ai_logs_dir = Path(__file__).parent.parent / 'ai_logs'
    
    all_results = []
    
    # Parse .log files
    log_files = list(ai_logs_dir.glob('*.log'))
    print(f"Parsing {len(log_files)} .log files...")
    
    for log_file in log_files:
        print(f"  - {log_file.name}")
        result = parse_log_file(log_file)
        all_results.append(result)
    
    # Parse .ndjson files
    ndjson_files = list(ai_logs_dir.glob('*.ndjson'))
    print(f"\nParsing {len(ndjson_files)} .ndjson files...")
    
    for ndjson_file in ndjson_files:
        print(f"  - {ndjson_file.name}")
        result = parse_ndjson_file(ndjson_file)
        all_results.append(result)
    
    output_path = ai_logs_dir / 'parsed_summary.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Parsed {len(all_results)} files")
    print(f"📊 Summary saved to: {output_path}")
    
    total_stats = defaultdict(int)
    for result in all_results:
        for key, val in result.get('stats', {}).items():
            total_stats[key] += val
    
    print("\n📈 Aggregate Statistics:")
    for key in sorted(total_stats.keys()):
        print(f"  {key}: {total_stats[key]}")


if __name__ == '__main__':
    main()
