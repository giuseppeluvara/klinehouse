"""Read-only audit of the frozen warmup and prospective paper journal.

An entirely rewritten database cannot be ruled out by a local hash chain.
The report distinguishes that limit from timing and arithmetic checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from research.risk_pilot import DAY, WEEK


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def audit(root,cache=None):
    bundle=json.loads((root/'bundle.json').read_text())
    for name,expected in bundle['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    for name,expected in bundle['runtime'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    config=json.loads((root/'protocol.json').read_text())
    sources=json.loads((root/'warmup-source-receipts.json').read_text())
    datasets=[]
    for receipt in sources:
        assert hashlib.sha256(receipt['raw_text'].encode()).hexdigest()==receipt['sha256']
        assert receipt['requested_ms']<=receipt['received_ms']
        rows=json.loads(receipt['raw_text'])
        datasets.append({int(x[0]):x for x in rows})
    history=json.loads((root/'warmup-history.json').read_text())
    predictions=json.loads((root/'warmup-predictions.json').read_text())
    assert len(history)==len(predictions)==52
    assert len({r['exit_ms'] for r in history})==52
    for row,pred in zip(history,predictions):
        anchor=row['anchor_ms']
        assert pred['anchor_ms']==anchor and row['exit_ms']==anchor+8*DAY
        assert row['exit_ms']<=min(s['requested_ms'] for s in sources)
        assert row['classification']=='retrospective warmup only'
        for asset,data in enumerate(datasets):
            context=[data[t] for t in range(anchor-257*DAY,anchor,DAY)]
            assert all(int(x[6])==int(x[0])+DAY-1 and int(x[6])<anchor for x in context)
            closes=[float(x[4]) for x in context]
            rates=[math.log(b/a)**2 for a,b in zip(closes,closes[1:])]
            weights=[math.exp(-math.log(2)*(255-i)/20) for i in range(256)]
            ewma=7*sum(w*r for w,r in zip(weights,rates))/sum(weights)
            variance=sum(math.exp(x) for x in pred['median_log_variances'][asset][1:8])
            opens=[float(data[anchor+i*DAY][1]) for i in range(1,9)]
            realized=sum(math.log(b/a)**2 for a,b in zip(opens,opens[1:]))
            assert abs(ewma-row['ewma'][asset])<1e-12
            assert abs(variance-row['timestral'][asset])<1e-12
            assert abs(realized-row['realized_variance'][asset])<1e-12
    reproduced=0
    max_error=0.
    if cache is not None:
        # Actual independent first/middle/last model inference, no outcome inputs.
        import numpy as np
        import torch
        from timestral import TimestralForecaster
        torch.set_num_threads(2)
        model=TimestralForecaster.from_pretrained(config['model']['id'],device='cpu',
            revision=config['model']['weight_revision'],cache_dir=cache,local_files_only=True)
        for index in (0,len(predictions)//2,len(predictions)-1):
            pred=predictions[index]
            a=pred['anchor_ms']
            closes=np.array([[float(data[t][4]) for t in range(a-257*DAY,a,DAY)] for data in datasets])
            r=np.diff(np.log(closes),axis=1)
            tensor=np.log(r*r+1e-8)[None].astype(np.float32)
            result=model.forecast(tensor,prediction_length=8,clip_nonnegative=False)
            med=result.quantiles[0,:,:,10].detach().cpu().numpy()
            error=float(np.max(np.abs(med-np.array(pred['median_log_variances']))))
            assert error<1e-6
            max_error=max(max_error,error)
            reproduced+=1
    evidence=dict(warmup_rows_verified=52,warmup_source_hashes_verified=len(sources),
        warmup_inferences_reproduced=reproduced,max_warmup_inference_error=max_error,
        classification='historical warmup; not prospective observations')
    path=root/'journal.sqlite'
    if not path.exists():
        evidence.update(journal_state='not activated',events=0,forecasts=0,paper_entries=0,outcomes=0)
        return evidence
    connection=sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro',uri=True)
    try:
        connection.row_factory=sqlite3.Row
        # A read transaction gives one consistent WAL snapshot while the writer runs.
        connection.execute('BEGIN')
        stored=json.loads(connection.execute("SELECT value FROM metadata WHERE key='config'").fetchone()[0])
        expected=dict(config,bundle_sha256=hashlib.sha256((root/'bundle.json').read_bytes()).hexdigest())
        assert stored==expected
        rows=list(connection.execute('SELECT * FROM events ORDER BY seq'))
    finally:connection.close()
    by_key={r['event_key']:r for r in rows}
    previous='0'*64
    forecasts=entries=outcomes=0
    missing=[]
    for seq,row in enumerate(rows,1):
        payload=json.loads(row['payload'])
        event=dict(seq=seq,event_key=row['event_key'],kind=row['kind'],recorded_ms=row['recorded_ms'],
                   payload=payload,previous=row['previous'])
        assert row['seq']==seq and row['previous']==previous and sha(event)==row['hash']
        previous=row['hash']
        if row['kind']=='source':
            assert hashlib.sha256(payload['raw_text'].encode()).hexdigest()==payload['sha256']
            assert payload['requested_ms']<=payload['received_ms']<=row['recorded_ms']
            assert payload['url'].startswith('https://api.binance.com/api/v3/')
        if row['kind']=='forecast':
            forecasts+=1
            a=payload['anchor_ms']
            assert config['first_anchor_ms']<=a<config['last_exit_ms']-DAY
            assert (a-config['first_anchor_ms'])%WEEK==0
            assert a<=row['recorded_ms']<=a+config['forecast_deadline_ms']
            assert payload['input_end_exclusive_ms']==a and payload['entry_ms']==a+DAY
            assert payload['calibration_max_label_end_ms']<=a
            for key in payload['source_keys']:
                source=by_key[key];receipt=json.loads(source['payload'])
                assert source['recorded_ms']<=row['recorded_ms']
                assert receipt['received_ms']<=row['recorded_ms']
                assert [x[0] for x in json.loads(receipt['raw_text'])]==list(range(a-257*DAY,a,DAY))
                assert all(x[6]<a for x in json.loads(receipt['raw_text']))
        if row['kind']=='forecast_ack':
            a=payload['anchor_ms'];forecast=by_key[f'decision:{a}']
            assert forecast['kind']=='forecast' and forecast['hash']==payload['forecast_hash']
            assert forecast['recorded_ms']<=row['recorded_ms']<=a+config['forecast_deadline_ms']
        if row['kind']=='entry':
            entries+=1
            a=payload['anchor_ms']
            assert by_key[f'decision:{a}']['kind']=='forecast' and f'ack:{a}' in by_key
            assert a+DAY<=row['recorded_ms']<=a+DAY+config['entry_deadline_ms']
            assert payload['earliest_source_requested_ms']>=a+DAY
            assert payload['latest_source_received_ms']<=row['recorded_ms']
            for wallet in payload['wallets'].values():
                assert wallet['cash']>=0 and min(wallet['units'])>=0
        if row['kind']=='outcome':
            outcomes+=1
            a=payload['anchor_ms']
            assert row['recorded_ms']>=a+8*DAY
            assert payload['calibration_row']['exit_ms']==a+8*DAY
        if row['kind'].startswith('missing') or row['kind']=='unacknowledged':
            missing.append(dict(kind=row['kind'],payload=payload))
    evidence.update(journal_state='expired' if 'expired' in by_key else 'activated',events=len(rows),
        head=previous,forecasts=forecasts,paper_entries=entries,outcomes=outcomes,
        missing_slots=missing,local_hash_chain_limitation='does not detect replacement of entire database without an external head receipt')
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--model-cache',type=Path)
    parser.add_argument('--save',type=Path)
    args=parser.parse_args()
    result=audit(args.root,args.model_cache)
    if args.save:
        with args.save.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
