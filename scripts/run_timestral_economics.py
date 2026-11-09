"""Run the registered Timestral economic diagnostic, retaining raw price data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import urllib.parse
import urllib.request

from research.timestral_economics import Trade, exposure, net_return, wealth

HOUR = 3600000


def timestamp(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000)


def quantile(values, probability):
    values = sorted(values)
    at = (len(values) - 1) * probability
    low = int(at)
    high = min(len(values) - 1, low + 1)
    return values[low] + (at - low) * (values[high] - values[low])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('reports/timestral-economics-20261104'))
    args = parser.parse_args()
    root = args.out
    if (root / 'results.json').exists():
        raise ValueError('completed result exists; do not overwrite')
    protocol = json.loads((root / 'protocol.json').read_text())
    source = Path(protocol['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol['source_sha256']:
        raise ValueError('forecast source changed')
    with gzip.open(source, 'rt') as handle:
        predictions = [json.loads(line) for line in handle]
    predictions = [p for p in predictions if p['condition'] == protocol['condition']]
    start, end = timestamp('2026-01-01T00:00:00Z'), timestamp('2026-08-01T00:00:00Z')
    cursor = start
    prices = {}
    receipts = []
    while cursor < end:
        url = 'https://api.binance.com/api/v3/klines?' + urllib.parse.urlencode(
            dict(symbol='BTCUSDT', interval='1h', startTime=cursor, endTime=end-1, limit=1000))
        request_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
        received_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        name = f'prices-{len(receipts):03d}.json'
        (root / name).write_bytes(raw)
        receipts.append(dict(url=url, file=name, requested_ms=request_ms,
                             received_ms=received_ms, sha256=hashlib.sha256(raw).hexdigest()))
        (root / 'price-manifest.json').write_text(json.dumps(receipts, indent=2))
        bars = json.loads(raw)
        if not bars:
            raise ValueError('empty price page')
        for bar in bars:
            t, price = int(bar[0]), float(bar[1])
            if t < start or t >= end or t % HOUR or t in prices or not math.isfinite(price) or price <= 0:
                raise ValueError('invalid, duplicate or outside-window price')
            prices[t] = price
        next_cursor = int(bars[-1][0]) + HOUR
        if next_cursor <= cursor:
            raise ValueError('nonadvancing pagination')
        cursor = next_cursor
    if set(prices) != set(range(start, end, HOUR)):
        raise ValueError('hourly source is incomplete')
    results = []
    trade_rows = []
    for horizon in protocol['horizons_hours']:
        anchors = sorted([p for p in predictions if p['horizon_hours'] == horizon],
                         key=lambda p: p['anchor'])
        ids = [p['sample_id'] for p in anchors]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate sample id')
        usable = [p for p in anchors if timestamp(p['anchor']) + horizon * HOUR < end]
        if len(usable) < 30:
            raise ValueError('insufficient forecast coverage')
        for left, right in zip(usable, usable[1:]):
            if timestamp(right['anchor']) - timestamp(left['anchor']) != horizon * HOUR:
                raise ValueError('missing or overlapping forecast anchors')
        for policy in protocol['policies']:
            weekly, weekly_benchmark, monthly = defaultdict(list), defaultdict(list), defaultdict(list)
            returns, benchmark, weights = [], [], []
            for p in usable:
                t = timestamp(p['anchor'])
                entry, exit = t + HOUR, t + horizon * HOUR
                pred = p['prediction']
                w = exposure(policy, pred['all_quantile_levels'], pred['all_return_quantiles'],
                             horizon, .001, .0002)
                trade = Trade(entry, exit, prices[entry], prices[exit], w)
                ret = net_return(trade, .001, .0002)
                bm = net_return(Trade(entry, exit, prices[entry], prices[exit], 1), .001, .0002)
                date = datetime.fromtimestamp(exit/1000, timezone.utc)
                iso = date.isocalendar()
                week = f'{iso.year}-{iso.week:02}'
                month = date.strftime('%Y-%m')
                weekly[week].append(ret)
                weekly_benchmark[week].append(bm)
                monthly[month].append(ret)
                returns.append(ret)
                benchmark.append(bm)
                weights.append(w)
                trade_rows.append(dict(horizon=horizon, policy=policy, sample_id=p['sample_id'],
                                       anchor=p['anchor'], entry_ms=entry, exit_ms=exit,
                                       entry_price=prices[entry], exit_price=prices[exit],
                                       exposure=w, net_return=ret, matched_benchmark_return=bm))
            weeks = sorted(weekly)
            # The same seed/ordered weeks preserves paired draws across all arms.
            rng = random.Random(protocol['bootstrap']['seed'])
            weekly_returns = [wealth(weekly[w])[0] for w in weeks]
            differences = [wealth(weekly[w])[0] - wealth(weekly_benchmark[w])[0] for w in weeks]
            boot_net, boot_excess = [], []
            for _ in range(protocol['bootstrap']['replicates']):
                draw = [rng.randrange(len(weeks)) for _ in weeks]
                boot_net.append(mean(weekly_returns[i] for i in draw))
                boot_excess.append(mean(differences[i] for i in draw))
            alpha = .05 / protocol['bootstrap']['family_size'] / 2
            ci_net = [quantile(boot_net, alpha), quantile(boot_net, 1-alpha)]
            ci_excess = [quantile(boot_excess, alpha), quantile(boot_excess, 1-alpha)]
            total, drawdown = wealth(returns)
            bm_total = wealth(benchmark)[0]
            month_returns = {m: wealth(r)[0] for m, r in sorted(monthly.items())}
            count = sum(w > 0 for w in weights)
            first, last = timestamp(usable[0]['anchor']) + HOUR, timestamp(usable[-1]['anchor']) + horizon * HOUR
            days = (last-first)/86400000
            buy_hold = net_return(Trade(first, last, prices[first], prices[last], 1), .001, .0002)
            cost_capital = .05 * days/365
            gate = (count >= 30 and total > 0 and total > bm_total and ci_net[0] > 0
                    and ci_excess[0] > 0 and sum(v > 0 for v in month_returns.values()) >= 5)
            results.append(dict(policy=policy, horizon=horizon, samples=len(usable),
                                excluded_boundary_samples=len(anchors)-len(usable), trades=count,
                                net_trading_return=total, max_drawdown_at_trade_exits=drawdown,
                                matched_always_long_return=bm_total, continuous_buy_hold_return=buy_hold,
                                capital_cost_fraction=cost_capital, economic_return=total-cost_capital,
                                mean_exposure=mean(weights), monthly_returns=month_returns,
                                weekly_net_ci_familywise=ci_net, weekly_excess_ci_familywise=ci_excess,
                                passes_historical_nomination_gate=gate))
    with (root / 'trades.jsonl').open('x') as handle:
        for row in trade_rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
    payload = dict(status='historical_exploration_only', protocol_sha256=hashlib.sha256(
        (root/'protocol.json').read_bytes()).hexdigest(), hourly_bars=len(prices),
        source_response_count=len(receipts), results=results,
        limitations=['Quantiles not certified calibrated probabilities',
                     'Delayed hourly opens are proxies, not measured fills',
                     'Spread/slippage is assumed; no historical L2',
                     'Drawdown observed at exits understates intratrade risk',
                     'Previously explored forecasts; pretraining overlap cannot be excluded',
                     'Separate horizons share returns; bootstrap is descriptive, not untouched evidence'])
    with (root / 'results.json').open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
