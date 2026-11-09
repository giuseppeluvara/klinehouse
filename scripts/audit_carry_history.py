"""Fixed-universe historical diagnostic; never selects coins on future funding.

Run from repository root with python -m scripts.audit_carry_history --out PATH.
Raw HTTP responses and hashes are preserved. Candle opens are price proxies,
not fills. No liquidation claim, prospective claim or orders.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

from research.carry_accounting import Costs, Prices, Settlement, account


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def get(host: str, path: str, **params):
        url = host + path + '?' + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
        name = f'raw-{len(manifest):04d}.json'
        (args.out / name).write_bytes(raw)
        manifest.append(dict(url=url, file=name, sha256=hashlib.sha256(raw).hexdigest()))
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        return json.loads(raw)

    spot = 'https://api.binance.com'
    perp = 'https://fapi.binance.com'
    # Closed UTC day, then three fixed, disjoint 30-day windows.
    server = get(spot, '/api/v3/time')['serverTime']
    end = server // 86400000 * 86400000
    start = end - 90 * 86400000
    protocol = dict(symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], start_ms=start,
                    end_ms=end, window_days=30, spot_notional='10000', capital='20000',
                    spot_fee='.001', perp_fee='.0005', slippage_bps_each_fill='2',
                    annual_capital_rate='.05', interpretation='historical diagnostic only; candle-open proxies; no liquidation model')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = []
    for symbol in protocol['symbols']:
        events = []
        cursor = start
        while cursor < end:
            rows = get(perp, '/fapi/v1/fundingRate', symbol=symbol,
                       startTime=cursor, endTime=end-1, limit=1000)
            if not rows:
                break
            events.extend(Settlement(int(r['fundingTime']), D(r['fundingRate']),
                                     D(r['markPrice'])) for r in rows)
            next_cursor = int(rows[-1]['fundingTime']) + 1
            if next_cursor <= cursor:
                raise ValueError('funding pagination did not advance')
            cursor = next_cursor
        if not events:
            raise ValueError('no funding data')

        def prices(timestamp: int) -> Prices:
            values = []
            for host, path in ((spot, '/api/v3/klines'), (perp, '/fapi/v1/klines')):
                bars = get(host, path, symbol=symbol, interval='1m',
                           startTime=timestamp, endTime=timestamp+59999, limit=1)
                if len(bars) != 1 or int(bars[0][0]) != timestamp:
                    raise ValueError('missing exact boundary candle')
                values.append(D(bars[0][1]))
            return Prices(timestamp, values[0], values[1])

        boundaries = [prices(start + i * 30 * 86400000) for i in range(4)]
        for entry, exit in zip(boundaries, boundaries[1:]):
            quantity = D(10000) / entry.spot
            result = account(entry, exit, quantity, events,
                             Costs(D('.001'), D('.0005'), D(2), D('.05')), D(20000))
            result.update(symbol=symbol, entry_ms=entry.time_ms, exit_ms=exit.time_ms)
            results.append(result)
    payload = dict(generated_at=datetime.now(timezone.utc).isoformat(),
                   protocol=protocol, results=results,
                   status='diagnostic_only',
                   limitations=['No intraperiod margin/liquidation simulation',
                                'No order book or actual fills',
                                'No funding completeness certification',
                                'Explored historical universe; not untouched validation'])
    (args.out / 'results.json').write_text(json.dumps(payload, indent=2, default=str))
    for row in results:
        print(row['symbol'], row['entry_ms'], 'net', round(row['net'], 2),
              'funding', round(row['funding'], 2), 'basis', round(row['basis_pnl'], 2))


if __name__ == '__main__':
    main()
