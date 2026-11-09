"""Run a frozen weekly Timestral risk-allocation experiment (public data only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import torch
from timestral import TimestralForecaster

from research.spot_portfolio import rebalance

DAY = 86400000


def ms(text):
    return int(datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()*1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('reports/timestral-risk-allocation-20261104'))
    parser.add_argument('--model-cache', type=Path, required=True)
    args = parser.parse_args()
    root = args.out
    if (root/'results.json').exists() or (root/'forecasts.jsonl').exists():
        raise ValueError('experiment already started; preserve its artifacts')
    protocol_bytes = (root/'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    start, end = ms(protocol['source_start']), ms(protocol['source_end_exclusive'])
    torch.set_num_threads(2)
    torch.manual_seed(7)
    installation = json.loads(importlib.metadata.distribution('timestral').read_text('direct_url.json'))
    if installation.get('vcs_info', {}).get('commit_id') != protocol['model']['code_revision']:
        raise ValueError('wrong Timestral source revision')
    # Verify actual checkpoint bytes, not merely a claimed model name.
    from huggingface_hub import hf_hub_download
    weight_file = Path(hf_hub_download(protocol['model']['id'], 'model.safetensors',
        revision=protocol['model']['weight_revision'], cache_dir=args.model_cache, local_files_only=True))
    if hashlib.sha256(weight_file.read_bytes()).hexdigest() != protocol['model']['weights_sha256']:
        raise ValueError('wrong Timestral weights')
    model = TimestralForecaster.from_pretrained(protocol['model']['id'],
        revision=protocol['model']['weight_revision'], cache_dir=args.model_cache,
        local_files_only=True, device='cpu')
    receipts, datasets, validity = [], [], []
    anomalies = []
    for symbol in protocol['universe']:
        cursor, rows = start, []
        while cursor < end:
            url = 'https://api.binance.com/api/v3/klines?' + urllib.parse.urlencode(dict(
                symbol=symbol, interval='1d', startTime=cursor, endTime=end-1, limit=1000))
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read()
            name = f'raw-{len(receipts):03d}.json'
            (root/name).write_bytes(raw)
            receipts.append(dict(url=url, file=name, sha256=hashlib.sha256(raw).hexdigest(),
                                 received_at=datetime.now(timezone.utc).isoformat()))
            (root/'source-manifest.json').write_text(json.dumps(receipts, indent=2))
            page = json.loads(raw)
            if not page:
                raise ValueError('empty source page')
            rows.extend(page)
            next_cursor = int(page[-1][0]) + DAY
            if next_cursor <= cursor:
                raise ValueError('source pagination did not advance')
            cursor = next_cursor
        times = [int(row[0]) for row in rows]
        if times != list(range(start, end, DAY)):
            raise ValueError('noncomplete or unordered daily source')
        complete = np.array([int(row[6]) == int(row[0]) + DAY - 1 for row in rows])
        for row, good in zip(rows, complete):
            if not good:
                anomalies.append(dict(symbol=symbol, open_time=row[0], close_time=row[6],
                                      reason='partial daily candle; excluded without imputation'))
        values = np.array([[float(row[1]), float(row[4])] for row in rows])
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('invalid daily prices')
        datasets.append(values)
        validity.append(complete)
    (root/'source-anomalies.json').write_text(json.dumps(anomalies,indent=2))
    opens = np.stack([x[:, 0] for x in datasets])
    closes = np.stack([x[:, 1] for x in datasets])
    returns = np.diff(np.log(closes), axis=1)
    complete_days = np.stack(validity).all(axis=0)
    complete_returns = complete_days[1:] & complete_days[:-1]
    times = np.arange(start, end, DAY)
    histories = []
    evaluated = []
    ewma_weights = np.exp(-math.log(2)*np.arange(255, -1, -1)/20)
    ewma_weights /= ewma_weights.sum()
    feature_file = (root/'forecasts.jsonl').open('x')
    try:
        for i in range(257, len(times)-8):
            anchor = int(times[i])
            if datetime.fromtimestamp(anchor/1000, timezone.utc).weekday() != 0:
                continue
            if not complete_returns[i-257:i-1].all() or not complete_days[i+1:i+9].all():
                continue
            context_r = returns[:, i-257:i-1]
            if context_r.shape != (2, 256):
                raise ValueError('wrong causal context shape')
            prediction = model.forecast(np.log(context_r**2 + 1e-8)[None].astype(np.float32),
                                       prediction_length=8, clip_nonnegative=False)
            median = prediction.quantiles[0, :, :, 10].detach().cpu().numpy().astype(float)
            if not np.isfinite(median).all():
                raise ValueError('nonfinite Timestral prediction')
            variance = np.exp(median[:, 1:8]).sum(axis=1)
            baseline = 7 * ((context_r**2) @ ewma_weights)
            if not np.isfinite(variance).all() or (variance <= 0).any():
                raise ValueError('nonpositive variance forecast')
            entry, exit = i+1, i+8
            mature = [x for x in histories if x['exit_ms'] <= anchor][-52:]
            scales = {}
            for method in ('timestral', 'ewma'):
                scales[method] = (np.mean([np.array(x['realized_variance']) / np.array(x[method])
                                          for x in mature], axis=0) if mature else np.ones(2))
            trend = np.log(closes[:, i-1] / closes[:, i-127]) > 0
            weights = {}
            for method, raw_var in (('timestral', variance), ('ewma', baseline)):
                sigma = np.sqrt(raw_var*scales[method]*365/7)
                weights[method] = np.minimum(.5, .1/sigma) * trend
            weights['fixed'] = .5 * trend.astype(float)
            row = dict(anchor_ms=anchor, input_end_exclusive_ms=anchor, entry_ms=int(times[entry]),
                       exit_ms=int(times[exit]), calibration_n=len(mature),
                       calibration_max_label_end_ms=max((x['exit_ms'] for x in mature), default=None),
                       timestral=variance.tolist(), ewma=baseline.tolist(),
                       calibration={k:v.tolist() for k,v in scales.items()},
                       trend=trend.tolist(), weights={k:v.tolist() for k,v in weights.items()},
                       median_log_variance_forecast=median.tolist())
            feature_file.write(json.dumps(row, allow_nan=False)+'\n')
            feature_file.flush()
            realized = np.sum(np.diff(np.log(opens[:, entry:exit+1]), axis=1)**2, axis=1)
            histories.append(dict(exit_ms=int(times[exit]), timestral=variance.tolist(),
                                  ewma=baseline.tolist(), realized_variance=realized.tolist()))
            if anchor >= ms(protocol['evaluation_start']) and len(mature) >= 26:
                evaluated.append(dict(**row, entry_index=entry, exit_index=exit))
            if len(histories) % 52 == 0:
                print('forecast weeks', len(histories), 'evaluation weeks', len(evaluated), flush=True)
    finally:
        feature_file.close()
    if len(evaluated) < 300:
        raise ValueError('insufficient evaluation weeks')
    for left, right in zip(evaluated, evaluated[1:]):
        if left['exit_ms'] != right['entry_ms']:
            raise ValueError('portfolio holding windows are not contiguous')
    results, all_returns, journals = {}, {}, []
    for arm in ('timestral', 'ewma', 'fixed', 'buy_hold'):
        cash, units, total_cost = 10000.0, [0.0, 0.0], 0.0
        weekly_returns, equity = [], [10000.0]
        yearly = {}
        for n, row in enumerate(evaluated):
            prices = opens[:, row['entry_index']].tolist()
            before = cash + sum(u*p for u,p in zip(units, prices))
            if arm != 'buy_hold' or n == 0:
                target = row['weights'][arm] if arm != 'buy_hold' else [.5, .5]
                cash, units, cost = rebalance(cash, units, prices, target, .001, .0002)
                total_cost += cost
            else:
                cost = 0.0
            exit_prices = opens[:, row['exit_index']].tolist()
            if n == len(evaluated)-1:
                cash, units, unwind = rebalance(cash, units, exit_prices, [0,0], .001, .0002)
                total_cost += unwind
                cost += unwind
            after = cash + sum(u*p for u,p in zip(units, exit_prices))
            ret = after/before - 1
            weekly_returns.append(ret)
            equity.append(after)
            year = datetime.fromtimestamp(row['entry_ms']/1000, timezone.utc).year
            yearly.setdefault(year, []).append(ret)
            journals.append(dict(arm=arm, entry_ms=row['entry_ms'], exit_ms=row['exit_ms'],
                                 nav_before=before, nav_after=after, net_return=ret,
                                 cash_after=cash, units_after=units, explicit_cost=cost))
        years = (evaluated[-1]['exit_ms'] - evaluated[0]['entry_ms'])/DAY/365
        nav = np.array(equity)
        max_dd = float(np.max(1 - nav / np.maximum.accumulate(nav)))
        results[arm] = dict(final_nav=float(nav[-1]), net_return=float(nav[-1]/10000-1),
            net_cagr=float((nav[-1]/10000)**(1/years)-1), explicit_cost_sum=total_cost,
            weekly_exit_max_drawdown=max_dd,
            yearly_returns={str(k):float(np.prod(1+np.array(v))-1) for k,v in yearly.items()})
        all_returns[arm] = np.array(weekly_returns)
    rng = np.random.default_rng(20261104)
    n = len(evaluated)
    indices = np.empty((10000,n), dtype=np.int32)
    indices[:,0] = rng.integers(n,size=10000)
    for t in range(1,n):
        restart = rng.random(10000) < 1/13
        indices[:,t] = np.where(restart,rng.integers(n,size=10000),(indices[:,t-1]+1)%n)
    for arm, values in all_returns.items():
        draws = values[indices].mean(axis=1)
        results[arm]['weekly_mean_ci99'] = np.quantile(draws,[.005,.995]).tolist()
        results[arm]['passes_profit_risk_screen'] = bool(results[arm]['net_cagr']>.05
            and results[arm]['weekly_mean_ci99'][0]>0 and results[arm]['weekly_exit_max_drawdown']<=.25)
    utility = lambda r: r - 1.5*r*r
    difference = utility(all_returns['timestral'])-utility(all_returns['ewma'])
    interval = np.quantile(difference[indices].mean(axis=1),[.005,.995]).tolist()
    payload = dict(status='historical_exploration_only', protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        evaluation_weeks=n, total_model_forecasts=len(histories), source_responses=len(receipts),
        first_entry_ms=evaluated[0]['entry_ms'], last_exit_ms=evaluated[-1]['exit_ms'],
        arms=results, paired_timestral_minus_ewma_weekly_utility_ci99=interval,
        timestral_incremental_value_screen=bool(interval[0]>0),
        versions={k:importlib.metadata.version(k) for k in ('timestral','torch','numpy','huggingface_hub')},
        limitations=protocol['limitations']+['Fractional quantities and historical daily-open execution proxies',
                     'Drawdown is marked weekly, not intraday', 'No real orders or prospective returns'])
    with (root/'portfolio-journal.jsonl').open('x') as f:
        for row in journals:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (root/'results.json').open('x') as f:json.dump(payload,f,indent=2,allow_nan=False)
    print(json.dumps(payload,indent=2))


if __name__ == '__main__':
    main()
