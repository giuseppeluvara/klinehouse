"""Independent algebraic portfolio, raw-data and causal calibration audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import urllib.parse

import numpy as np

DAY = 86400000


def target_portfolio(cash, units, prices, weights):
    """Closed-form solution per buy/sell region, independent of bisection engine."""
    buy = 1.0002*1.001-1
    sell = 1-.9998*.999
    old = [u*p for u,p in zip(units,prices)]
    nav = cash+sum(old)
    for signs in itertools.product((False,True),repeat=2):
        slope = sum(w*(buy if side else -sell) for w,side in zip(weights,signs))
        intercept = sum(n*(-buy if side else sell) for n,side in zip(old,signs))
        after = (nav-intercept)/(1+slope)
        delta = [w*after-n for w,n in zip(weights,old)]
        if all((d>=-1e-8 if side else d<=1e-8) for d,side in zip(delta,signs)):
            return after*(1-sum(weights)), [w*after/p for w,p in zip(weights,prices)], nav-after
    raise AssertionError('no consistent analytic cost region')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model-cache',type=Path,required=True)
    args=parser.parse_args()
    root=args.out
    results=json.loads((root/'results.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==results['protocol_sha256']
    for file,digest in json.loads((root/'code-manifest.json').read_text()).items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    data={s:{} for s in protocol['universe']}
    manifest=json.loads((root/'source-manifest.json').read_text())
    for receipt in manifest:
        raw=(root/receipt['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
        params=urllib.parse.parse_qs(urllib.parse.urlparse(receipt['url']).query)
        symbol=params['symbol'][0]
        for row in json.loads(raw):
            assert row[0]<1785542400000
            assert row[0] not in data[symbol]
            data[symbol][row[0]]=row
    forecasts=[json.loads(x) for x in (root/'forecasts.jsonl').read_text().splitlines()]
    history=[]
    checked=0
    for row in forecasts:
        anchor=row['anchor_ms']
        assert row['entry_ms']==anchor+DAY and row['exit_ms']==anchor+8*DAY
        assert row['input_end_exclusive_ms']==anchor
        eligible=[x for x in history if x['end']<=anchor][-52:]
        assert len(eligible)==row['calibration_n']
        assert row['calibration_max_label_end_ms']==max((x['end'] for x in eligible),default=None)
        realized=[]
        for asset,symbol in enumerate(protocol['universe']):
            inputs=[data[symbol][anchor-i*DAY] for i in range(257,0,-1)]
            assert all(int(x[6])==int(x[0])+DAY-1 for x in inputs)
            squared=[math.log(float(b[4])/float(a[4]))**2 for a,b in zip(inputs,inputs[1:])]
            weights=[math.exp(-math.log(2)*(255-i)/20) for i in range(256)]
            ewma=7*sum(w*r for w,r in zip(weights,squared))/sum(weights)
            assert abs(ewma-row['ewma'][asset])<1e-12
            forecast=sum(math.exp(x) for x in row['median_log_variance_forecast'][asset][1:8])
            assert abs(forecast-row['timestral'][asset])<1e-12
            trend=math.log(float(inputs[-1][4])/float(inputs[-127][4]))>0
            assert trend==row['trend'][asset]
            for method in ('ewma','timestral'):
                scale=(sum(x['rv'][asset]/x[method][asset] for x in eligible)/len(eligible)
                       if eligible else 1)
                assert abs(scale-row['calibration'][method][asset])<1e-9
                weight=min(.5,.1/math.sqrt(row[method][asset]*scale*365/7))*trend
                assert abs(weight-row['weights'][method][asset])<1e-10
            opens=[float(data[symbol][row['entry_ms']+i*DAY][1]) for i in range(8)]
            realized.append(sum(math.log(b/a)**2 for a,b in zip(opens,opens[1:])))
        history.append(dict(end=row['exit_ms'],rv=realized,timestral=row['timestral'],ewma=row['ewma']))
        checked+=1
    journal=[json.loads(x) for x in (root/'portfolio-journal.jsonl').read_text().splitlines()]
    by_entry={r['entry_ms']:r for r in forecasts}
    audits={}
    for arm in results['arms']:
        subset=[r for r in journal if r['arm']==arm]
        cash,units=10000.,[0.,0.]
        close_peak=range_peak=10000.
        close_dd=range_dd=0.
        total_cost=0.
        for n,record in enumerate(subset):
            entry,exit=record['entry_ms'],record['exit_ms']
            prices=[float(data[s][entry][1]) for s in protocol['universe']]
            nav_before=cash+sum(u*p for u,p in zip(units,prices))
            assert abs(nav_before-record['nav_before'])<1e-6
            if arm!='buy_hold' or n==0:
                weights=by_entry[entry]['weights'][arm] if arm!='buy_hold' else [.5,.5]
                cash,units,cost=target_portfolio(cash,units,prices,weights)
            else:cost=0.
            for t in range(entry,exit,DAY):
                closes=[float(data[s][t][4]) for s in protocol['universe']]
                lows=[float(data[s][t][3]) for s in protocol['universe']]
                highs=[float(data[s][t][2]) for s in protocol['universe']]
                close_nav=cash+sum(u*p for u,p in zip(units,closes))
                low_nav=cash+sum(u*p for u,p in zip(units,lows))
                high_nav=cash+sum(u*p for u,p in zip(units,highs))
                close_peak=max(close_peak,close_nav)
                range_peak=max(range_peak,high_nav)
                close_dd=max(close_dd,1-close_nav/close_peak)
                range_dd=max(range_dd,1-low_nav/range_peak)
            exit_prices=[float(data[s][exit][1]) for s in protocol['universe']]
            if n==len(subset)-1:
                cash,units,unwind=target_portfolio(cash,units,exit_prices,[0.,0.])
                cost+=unwind
            nav=cash+sum(u*p for u,p in zip(units,exit_prices))
            assert abs(nav-record['nav_after'])<1e-6
            assert abs(cash-record['cash_after'])<1e-6
            assert max(abs(a-b) for a,b in zip(units,record['units_after']))<1e-8
            assert abs(cost-record['explicit_cost'])<1e-6
            total_cost+=cost
        assert abs(total_cost-results['arms'][arm]['explicit_cost_sum'])<1e-5
        audits[arm]=dict(final_nav_verified=nav,daily_close_max_drawdown=close_dd,
                        conservative_daily_range_drawdown_bound=range_dd,
                        interpretation='range bound assumes simultaneous asset extremes and adverse intraday ordering; it is not an observed drawdown')
    # Re-run actual pinned inference on first/middle/last saved contexts.
    import torch
    from timestral import TimestralForecaster
    torch.set_num_threads(2)
    model=TimestralForecaster.from_pretrained(protocol['model']['id'],device='cpu',
        revision=protocol['model']['weight_revision'],cache_dir=args.model_cache,local_files_only=True)
    max_model_error=0.
    for index in (0,len(forecasts)//2,len(forecasts)-1):
        row=forecasts[index]
        context=[]
        for symbol in protocol['universe']:
            prices=[float(data[symbol][row['anchor_ms']-i*DAY][4]) for i in range(257,0,-1)]
            rr=np.diff(np.log(np.array(prices)))
            context.append(np.log(rr*rr+1e-8))
        pred=model.forecast(np.array(context,dtype=np.float32)[None],prediction_length=8,clip_nonnegative=False)
        actual=pred.quantiles[0,:,:,10].detach().cpu().numpy()
        error=float(np.max(np.abs(actual-np.array(row['median_log_variance_forecast']))))
        max_model_error=max(max_model_error,error)
        assert error<1e-6
    output=dict(status='passed',raw_response_hashes_verified=len(manifest),
                causal_forecast_rows_verified=checked,portfolio_rows_verified=len(journal),
                actual_model_inferences_reproduced=3,max_inference_error=max_model_error,
                portfolio_audits=audits,verified_at=datetime.now(timezone.utc).isoformat())
    with (root/'independent-audit.json').open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
