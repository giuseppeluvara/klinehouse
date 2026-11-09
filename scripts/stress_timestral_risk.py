"""Descriptive cost and entry-date stress of unchanged historical decisions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import urllib.parse

from research.spot_portfolio import rebalance

DAY=86400000
ARMS=('timestral','ewma','fixed','buy_hold')


def replay(frames,arm,fee,slippage):
    cash,units=10000.,[0.,0.]
    peak=10000.
    drawdown=0.
    cost_sum=0.
    year_end={}
    weekly=[]
    for n,frame in enumerate(frames):
        before=cash+sum(u*p for u,p in zip(units,frame['entry_prices']))
        if arm!='buy_hold' or n==0:
            weights=frame['weights'][arm] if arm!='buy_hold' else [.5,.5]
            cash,units,cost=rebalance(cash,units,frame['entry_prices'],weights,fee,slippage)
            cost_sum+=cost
        value=cash+sum(u*p for u,p in zip(units,frame['entry_prices']))
        peak=max(peak,value)
        drawdown=max(drawdown,1-value/peak)
        for t,prices in frame['daily_closes']:
            value=cash+sum(u*p for u,p in zip(units,prices))
            peak=max(peak,value)
            drawdown=max(drawdown,1-value/peak)
            year=datetime.fromtimestamp(t/1000,timezone.utc).year
            year_end[year]=value
        if n==len(frames)-1:
            cash,units,cost=rebalance(cash,units,frame['exit_prices'],[0.,0.],fee,slippage)
            cost_sum+=cost
        after=cash+sum(u*p for u,p in zip(units,frame['exit_prices']))
        peak=max(peak,after)
        drawdown=max(drawdown,1-after/peak)
        weekly.append(after/before-1)
    years=(frames[-1]['exit_ms']-frames[0]['entry_ms'])/DAY/365
    last_year=datetime.fromtimestamp(frames[-1]['exit_ms']/1000,timezone.utc).year
    year_end[last_year]=after
    previous=10000.
    yearly={}
    for year,value in sorted(year_end.items()):
        yearly[str(year)]=value/previous-1
        previous=value
    return dict(final_nav=after,net_return=after/10000-1,cagr=(after/10000)**(1/years)-1,
                daily_mark_max_drawdown=drawdown,explicit_execution_cost_sum=cost_sum,
                calendar_returns=yearly,calendar_end_nav=year_end,weeks=len(frames),
                start_ms=frames[0]['entry_ms'],end_ms=frames[-1]['exit_ms'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('reports/timestral-risk-stress-20261104'))
    args=parser.parse_args()
    root=args.out
    if (root/'results.json').exists():raise ValueError('retain existing completed diagnostic')
    protocol=json.loads((root/'protocol.json').read_text())
    source=Path(protocol['source'])
    assert hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest()==protocol['source_protocol_sha256']
    datasets={s:{} for s in ('BTCUSDT','ETHUSDT')}
    raw_count=0
    for receipt in json.loads((source/'source-manifest.json').read_text()):
        raw=(source/receipt['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
        symbol=urllib.parse.parse_qs(urllib.parse.urlparse(receipt['url']).query)['symbol'][0]
        for bar in json.loads(raw):datasets[symbol][int(bar[0])]=bar
        raw_count+=1
    forecasts={p['entry_ms']:p for p in map(json.loads,(source/'forecasts.jsonl').read_text().splitlines())}
    trades=[p for p in map(json.loads,(source/'portfolio-journal.jsonl').read_text().splitlines()) if p['arm']=='timestral']
    frames=[]
    for trade in trades:
        entry,exit=trade['entry_ms'],trade['exit_ms']
        assert exit-entry==7*DAY
        frames.append(dict(entry_ms=entry,exit_ms=exit,weights=forecasts[entry]['weights'],
            entry_prices=[float(data[entry][1]) for data in datasets.values()],
            exit_prices=[float(data[exit][1]) for data in datasets.values()],
            daily_closes=[(t+DAY-1,[float(data[t][4]) for data in datasets.values()]) for t in range(entry,exit,DAY)]))
    original=json.loads((source/'results.json').read_text())
    costs=[]
    base={}
    for multiplier in protocol['cost_multipliers']:
        for arm in ARMS:
            result=replay(frames,arm,.001*multiplier,.0002*multiplier)
            result.update(arm=arm,cost_multiplier=multiplier)
            costs.append(result)
            if multiplier==1:
                assert abs(result['final_nav']-original['arms'][arm]['final_nav'])<1e-6
                base[arm]=result
        print('cost scenario',multiplier,'completed',flush=True)
    windows=[]
    summaries=[]
    for length in (52,156):
        for arm in ARMS:
            series=[]
            for i in range(len(frames)-length+1):
                row=replay(frames[i:i+length],arm,.001,.0002)
                row.update(arm=arm,window_weeks=length)
                windows.append(row)
                series.append(row)
            summaries.append(dict(arm=arm,window_weeks=length,windows=len(series),
                positive_net_windows=sum(x['net_return']>0 for x in series),
                over_5pct_cagr_windows=sum(x['cagr']>.05 for x in series),
                within_25pct_drawdown_windows=sum(x['daily_mark_max_drawdown']<=.25 for x in series),
                joint_profit_and_risk_windows=sum(x['net_return']>0 and x['daily_mark_max_drawdown']<=.25 for x in series),
                minimum_net_return=min(x['net_return'] for x in series),
                maximum_net_return=max(x['net_return'] for x in series),
                minimum_cagr=min(x['cagr'] for x in series),
                maximum_cagr=max(x['cagr'] for x in series),
                worst_drawdown=max(x['daily_mark_max_drawdown'] for x in series)))
            print('rolling windows',length,arm,'completed',flush=True)
    attribution={}
    for arm,result in base.items():
        after2020=result['final_nav']/result['calendar_end_nav'][2020]-1
        years=(result['end_ms']-int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000))/DAY/365
        attribution[arm]=dict(calendar_returns=result['calendar_returns'],
            net_factor_excluding_2020=1+after2020,
            annualized_factor_excluding_2020=(1+after2020)**(1/years)-1,
            interpretation='multiplicative attribution of existing path; not a new cash-start strategy')
    output=dict(classification=protocol['classification'],raw_source_hashes_verified=raw_count,
        frozen_decisions_used=len(frames),cost_scenarios=costs,rolling_summary=summaries,
        attribution=attribution,limitations=['Overlapping windows are not independent trials or probabilities',
            'Fixed historical signal weights and calibration are not refitted within each window',
            'Transaction cost scenarios are assumptions, not measured fills',
            'Daily marks can miss intraday drawdowns','No new prospective results or live changes'])
    with (root/'rolling-windows.jsonl').open('x') as f:
        for row in windows:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (root/'results.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(dict(rolling_summary=summaries,attribution=attribution),indent=2))


if __name__=='__main__':main()
