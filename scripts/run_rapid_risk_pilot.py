"""One prospective 24-hour paper comparison using public Binance data."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from urllib.error import URLError

from research.risk_pilot import DAY, Journal, digest, paper_rebalance
from scripts.audit_risk_pilot_accounting import audit_accounting
from scripts.run_risk_pilot import capture, check_clock, load_bundle, load_model, public_get, quotes

HOUR = DAY // 24
BAR = 4 * HOUR
ARMS = ('timestral', 'ewma')
CALIBRATION_DAYS = 28
MARK_WINDOW = 10 * 60_000
COMMIT_MARGIN_MS = 1_000


def now_ms():
    return time.time_ns() // 1_000_000


def prices(receipt, start, end, step):
    rows = json.loads(receipt['raw_text'])
    if [int(row[0]) for row in rows] != list(range(start, end, step)):
        raise ValueError('incomplete candle sequence')
    for row in rows:
        low, high = float(row[3]), float(row[2])
        opening, close = float(row[1]), float(row[4])
        if (len(row) != 12 or int(row[6]) != int(row[0]) + step - 1 or
                not all(math.isfinite(v) and v > 0 for v in (low, high, opening, close)) or
                not low <= min(opening, close) <= max(opening, close) <= high):
            raise ValueError('invalid closed candle')
    return [float(row[4]) for row in rows]


def raw_variance(contexts, predict):
    medians = predict(contexts)
    decay = [2 ** ((i - 255) / 120) for i in range(256)]
    timestral = [sum(math.exp(x) for x in asset[:6]) for asset in medians]
    ewma = []
    for asset in contexts:
        squared = [math.log(b / a) ** 2 for a, b in zip(asset, asset[1:])]
        ewma.append(6 * sum(w * r for w, r in zip(decay, squared)) / sum(decay))
    if any(not math.isfinite(v) or v <= 0 for v in timestral + ewma):
        raise ValueError('invalid daily risk forecast')
    return dict(timestral=timestral, ewma=ewma)


def decision(config, journal, cache):
    clock = check_clock()
    clock_key = capture(journal, clock, 'rapid forecast clock')
    observed = clock['received_ms']
    bar_end = observed // BAR * BAR
    daily_end = observed // DAY * DAY
    start = bar_end - (257 + CALIBRATION_DAYS * 6) * BAR
    history, trend, source_keys = [], [], [clock_key]
    for symbol in config['symbols']:
        receipt = public_get('/api/v3/klines', symbol=symbol, interval='4h', startTime=start,
                             endTime=bar_end - 1, limit=257 + CALIBRATION_DAYS * 6)
        history.append(prices(receipt, start, bar_end, BAR))
        source_keys.append(capture(journal, receipt, 'rapid four-hour context and matured calibration'))
        daily_start = daily_end - 127 * DAY
        receipt = public_get('/api/v3/klines', symbol=symbol, interval='1d', startTime=daily_start,
                             endTime=daily_end - 1, limit=127)
        closes = prices(receipt, daily_start, daily_end, DAY)
        trend.append(closes[-1] > closes[0])
        source_keys.append(capture(journal, receipt, 'rapid shared 126-day trend filter'))
    predict, hashes = load_model(config, cache)
    if hashes != config['model_hashes']:
        raise ValueError('rapid model hashes changed')
    ratios = {arm: [[], []] for arm in ARMS}
    for offset in range(0, CALIBRATION_DAYS * 6, 6):
        split = offset + 257
        raw = raw_variance([asset[offset:split] for asset in history], predict)
        for index, asset in enumerate(history):
            target = asset[split - 1:split + 6]
            realized = sum(math.log(b / a) ** 2 for a, b in zip(target, target[1:]))
            for arm in ARMS:
                ratios[arm][index].append(realized / raw[arm][index])
    scales = {arm: [sum(asset) / CALIBRATION_DAYS for asset in ratios[arm]] for arm in ARMS}
    raw = raw_variance([asset[-257:] for asset in history], predict)
    weights = {}
    for arm in ARMS:
        calibrated = [scale * variance for scale, variance in zip(scales[arm], raw[arm])]
        if any(not math.isfinite(v) or v <= 0 for v in calibrated):
            raise ValueError('invalid calibrated daily risk')
        weights[arm] = [min(.5, .1 / math.sqrt(v * 365)) if up else 0.0
                        for v, up in zip(calibrated, trend)]
    if now_ms() + COMMIT_MARGIN_MS > config['entry_deadline_ms']:
        return
    journal.append(f"decision:{config['activation_ms']}", 'rapid_forecast', dict(
        source_keys=source_keys, weights=weights, trend=trend, raw_variance=raw,
        calibration=scales, calibration_windows=CALIBRATION_DAYS, context_end_ms=bar_end))


def initial_wallets(config):
    capital = config['initial_paper_capital_per_arm']
    return {arm: dict(cash=capital, units=[0., 0.], peak=capital, halted=False) for arm in ARMS}


def guarded_quotes(root, config, journal, label):
    if shutil.disk_usage(root).free < 40 * 1024 ** 3:
        raise RuntimeError('less than 40 GiB available')
    capture(journal, check_clock(), label + ' clock')
    return quotes(config, journal, label)


def enter(root, config, journal, forecast):
    books, metadata = guarded_quotes(root, config, journal, 'rapid paper entry')
    if metadata['earliest_source_requested_ms'] < forecast['recorded_ms']:
        raise ValueError('paper quotes precede the durable forecast')
    wallets = initial_wallets(config)
    fills = {arm: paper_rebalance(wallet['cash'], wallet['units'], books,
                                 forecast['payload']['weights'][arm]) for arm, wallet in wallets.items()}
    for arm, wallet in wallets.items():
        wallet.update(cash=fills[arm]['cash'], units=fills[arm]['units'])
    if now_ms() + COMMIT_MARGIN_MS <= config['entry_deadline_ms']:
        journal.append('entry', 'rapid_entry', dict(anchor_ms=config['activation_ms'],
            forecast_hash=forecast['hash'], wallets=wallets, fills=fills, **metadata))


def portfolio_mark(journal, key, wallets, books, metadata, final=False):
    marks = {}
    for arm, wallet in wallets.items():
        liquidation = paper_rebalance(wallet['cash'], wallet['units'], books, [0., 0.])
        nav = liquidation['cash']
        peak = max(wallet['peak'], nav)
        drawdown = 1 - nav / peak
        risk_halt = drawdown >= .25
        close = risk_halt or final
        wallet.update(peak=peak, halted=wallet['halted'] or close)
        if close:
            wallet.update(cash=nav, units=[0., 0.])
        marks[arm] = dict(liquidation_nav=nav, observed_drawdown=drawdown,
                          risk_halt=risk_halt, simulated_liquidation=liquidation if close else None)
    return journal.append(key, 'final_exit' if final else 'mark', dict(wallets=wallets, marks=marks, **metadata))


def accounting(journal, config):
    events = []
    kinds = {'rapid_forecast': 'forecast', 'rapid_entry': 'entry'}
    for row in journal.connection.execute('SELECT * FROM events ORDER BY seq'):
        event = dict(row)
        event['payload'] = json.loads(event['payload'])
        event['kind'] = kinds.get(event['kind'], event['kind'])
        events.append(event)
    return audit_accounting(events, config)


def result(journal, config, entry, exit_event):
    audit = accounting(journal, config)
    marks = [json.loads(row[0])['marks'] for row in journal.connection.execute(
        "SELECT payload FROM events WHERE kind IN ('mark','final_exit') ORDER BY seq")]
    initial = config['initial_paper_capital_per_arm']
    portfolios = {}
    for arm in ARMS:
        fees = audit['arms'][arm]['cumulative_simulated_fees']
        pnl = audit['arms'][arm]['cash'] - initial
        portfolios[arm] = dict(net_pnl_usdt=pnl, net_return_pct=100 * pnl / initial,
            observed_max_drawdown_pct=100 * max(mark[arm]['observed_drawdown'] for mark in marks),
            risk_halted=any(mark[arm]['risk_halt'] for mark in marks), simulated_fees_usdt=fees,
            entry_units=entry['payload']['wallets'][arm]['units'])
    quote_ms = exit_event['payload']['latest_source_received_ms']
    due = entry['recorded_ms'] + DAY
    context_end = journal.get(f"decision:{config['activation_ms']}")['payload']['context_end_ms']
    return dict(status='valued', entry_ms=entry['recorded_ms'], due_ms=due, quote_ms=quote_ms,
        context_end_ms=context_end, exit_lateness_ms=quote_ms - due, late_exit=quote_ms - due > MARK_WINDOW,
        marks_observed=len(marks) - 2, marks_expected=23,
        arms=portfolios, timestral_minus_ewma_pp=portfolios['timestral']['net_return_pct'] - portfolios['ewma']['net_return_pct'],
        accounting=audit, **journal.verify())


def step(root, config, journal, cache):
    entry = journal.get('entry')
    if entry is None:
        if now_ms() > config['entry_deadline_ms']:
            missed = journal.get('missing_entry') or journal.append('missing_entry', 'missing_entry',
                {'reason': 'ingresso non registrato entro la finestra iniziale'})
            return dict(status='missing_entry', **missed['payload'], **journal.verify())
        if shutil.disk_usage(root).free < 40 * 1024 ** 3:
            raise RuntimeError('less than 40 GiB available')
        forecast = journal.get(f"decision:{config['activation_ms']}")
        if forecast is None:
            decision(config, journal, cache)
        else:
            enter(root, config, journal, forecast)
        return None
    final = journal.get('final_exit')
    if final:
        return result(journal, config, entry, final)
    if journal.get('mark:0') is None:
        payload = entry['payload']
        books = [json.loads(journal.get(key)['payload']['raw_text']) for key in payload['source_keys']]
        metadata = {key: payload[key] for key in ('source_keys', 'earliest_source_requested_ms', 'latest_source_received_ms')}
        portfolio_mark(journal, 'mark:0', payload['wallets'], books, metadata)
        return None
    elapsed = now_ms() - entry['recorded_ms']
    slot = elapsed // HOUR
    final_due = elapsed >= DAY
    if not final_due and (slot == 0 or elapsed % HOUR > MARK_WINDOW or journal.get(f'mark:{slot}')):
        return None
    row = journal.connection.execute("SELECT payload FROM events WHERE kind='mark' ORDER BY seq DESC LIMIT 1").fetchone()
    wallets = json.loads(row[0])['wallets']
    label = 'rapid paper final exit' if final_due else 'rapid hourly risk mark'
    books, metadata = guarded_quotes(root, config, journal, label)
    if not final_due and now_ms() > entry['recorded_ms'] + slot * HOUR + MARK_WINDOW:
        return None
    portfolio_mark(journal, 'final_exit' if final_due else f'mark:{slot}', wallets, books, metadata, final_due)
    return None


def serve(root, cache):
    config = load_bundle(root)
    if config['orders'] is not False or config['maximum_observed_drawdown'] != .25:
        raise ValueError('unexpected rapid pilot risk configuration')
    config['model_hashes'] = json.loads((root / 'bundle.json').read_text())['model_hashes']
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(root / 'journal.sqlite', config)
        try:
            if journal.get('activation') is None:
                journal.append('activation', 'activation', {'protocol_sha256': digest(config)})
            while True:
                try:
                    report = step(root, config, journal, cache)
                except (URLError, TimeoutError) as exc:
                    print(json.dumps({'state': 'source_retry', 'at_ms': now_ms(), 'error': str(exc)}), flush=True)
                    time.sleep(10)
                    continue
                if report is not None:
                    temp = root / 'result.tmp'
                    with temp.open('w') as handle:
                        json.dump(report, handle, indent=2, allow_nan=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                    temp.replace(root / 'result.json')
                    print(json.dumps(report), flush=True)
                    return
                time.sleep(2)
        finally:
            journal.connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-cache', type=Path, required=True)
    args = parser.parse_args()
    serve(args.root, args.model_cache)


if __name__ == '__main__':
    main()
