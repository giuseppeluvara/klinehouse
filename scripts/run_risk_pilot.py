"""Bounded public-data, fractional-paper Timestral trial. Never sends orders."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import ssl
import time
import urllib.parse
import urllib.error
import urllib.request

from research.risk_pilot import ARMS, DAY, WEEK, Journal, canonical, closed_context, digest, forecast_weights, paper_rebalance

RUNTIME_FILES=('research/risk_pilot.py','scripts/run_risk_pilot.py')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('public data redirect refused')


def public_get(path,**params):
    if path not in ('/api/v3/time','/api/v3/klines','/api/v3/depth'):
        raise ValueError('unregistered public endpoint')
    url='https://api.binance.com'+path+'?'+urllib.parse.urlencode(params)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect,
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    before=time.time_ns()//1000000
    request=urllib.request.Request(url,headers={'Cache-Control':'no-cache'})
    with opener.open(request,timeout=3) as response:
        raw=response.read(1000001)
    after=time.time_ns()//1000000
    if len(raw)>1000000:
        raise ValueError('oversized public response')
    if after-before>3000:
        raise TimeoutError('public response exceeded3seconds')
    return dict(url=url,requested_ms=before,received_ms=after,
                raw_text=raw.decode('utf-8'),sha256=hashlib.sha256(raw).hexdigest())


def load_model(config,cache):
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from timestral import TimestralForecaster
    actual=json.loads(importlib.metadata.distribution('timestral').read_text('direct_url.json'))
    if actual.get('vcs_info',{}).get('commit_id')!=config['model']['code_revision']:
        raise ValueError('Timestral code revision changed')
    hashes={}
    for name in ('model.safetensors','config.json'):
        path=Path(hf_hub_download(config['model']['id'],name,revision=config['model']['weight_revision'],
                                 cache_dir=cache,local_files_only=True))
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    if hashes['model.safetensors']!=config['model']['weights_sha256']:
        raise ValueError('Timestral weight bytes changed')
    if hashes['config.json']!='b5f27f61513f8fd1a3aa996f3720d3679cb40bce5bcb08eb3dc320ce726e7c2f':
        raise ValueError('Timestral model configuration changed')
    torch.set_num_threads(2)
    model=TimestralForecaster.from_pretrained(config['model']['id'],device='cpu',
        revision=config['model']['weight_revision'],cache_dir=cache,local_files_only=True)

    def predict(closes):
        returns=np.diff(np.log(np.array(closes,dtype=float)),axis=1)
        context=np.log(returns**2+1e-8)[None].astype(np.float32)
        result=model.forecast(context,prediction_length=8,clip_nonnegative=False)
        return result.quantiles[0,:,:,10].detach().cpu().numpy().astype(float).tolist()
    return predict,hashes


def prepare(root,cache):
    if (root/'bundle.json').exists():raise ValueError('bundle already frozen')
    config=json.loads((root/'protocol.json').read_text())
    if time.time()*1000>=config['first_anchor_ms']:
        raise ValueError('cannot freeze this prospective trial after its first anchor')
    predictor,model_hashes=load_model(config,cache)
    now=int(time.time()*1000)
    closed_end=now//DAY*DAY
    today=datetime.fromtimestamp(now/1000,timezone.utc)
    monday=(now//DAY-today.weekday())*DAY
    last_anchor=monday
    while last_anchor+8*DAY>=closed_end:last_anchor-=WEEK
    anchors=[last_anchor-i*WEEK for i in range(51,-1,-1)]
    begin=anchors[0]-257*DAY
    receipts=[]
    by_symbol={}
    for symbol in config['symbols']:
        receipt=public_get('/api/v3/klines',symbol=symbol,interval='1d',startTime=begin,
                           endTime=closed_end-1,limit=1000)
        bars=json.loads(receipt['raw_text'])
        if [int(x[0]) for x in bars]!=list(range(begin,closed_end,DAY)):
            raise ValueError('warmup source incomplete')
        by_symbol[symbol]={int(x[0]):x for x in bars}
        receipts.append(receipt)
    warmup=[]
    predictions=[]
    for anchor in anchors:
        closes=[closed_context([by_symbol[s][t] for t in range(anchor-257*DAY,anchor,DAY)],anchor)
                for s in config['symbols']]
        median=predictor(closes)
        variance=[sum(math.exp(x) for x in asset[1:8]) for asset in median]
        decay=[math.exp(-math.log(2)*(255-i)/20) for i in range(256)]
        ewma=[]
        realized=[]
        for s,asset in zip(config['symbols'],closes):
            r2=[math.log(b/a)**2 for a,b in zip(asset,asset[1:])]
            ewma.append(7*sum(w*r for w,r in zip(decay,r2))/sum(decay))
            opens=[float(by_symbol[s][anchor+i*DAY][1]) for i in range(1,9)]
            realized.append(sum(math.log(b/a)**2 for a,b in zip(opens,opens[1:])))
        warmup.append(dict(anchor_ms=anchor,exit_ms=anchor+8*DAY,
            timestral=variance,ewma=ewma,realized_variance=realized,classification='retrospective warmup only'))
        predictions.append(dict(anchor_ms=anchor,median_log_variances=median))
    if len(warmup)!=52 or warmup[-1]['exit_ms']>now:
        raise ValueError('warmup did not mature before activation')
    for name,value in (('warmup-history.json',warmup),('warmup-predictions.json',predictions),
                       ('warmup-source-receipts.json',receipts)):
        with (root/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    # Readiness is off-schedule and must never be eligible as a strategy decision.
    context_anchor=closed_end
    closes=[closed_context([by_symbol[s][t] for t in range(context_anchor-257*DAY,context_anchor,DAY)],context_anchor)
            for s in config['symbols']]
    readiness=forecast_weights(closes,predictor(closes),warmup,context_anchor)
    readiness.update(classification='off-schedule readiness diagnostic; not a forecast or paper entry',
                     anchor_ms=context_anchor,actual_observed_ms=int(time.time()*1000))
    (root/'readiness.json').write_text(json.dumps(readiness,indent=2))
    files=['protocol.json','warmup-history.json','warmup-predictions.json','warmup-source-receipts.json']
    bundle=dict(frozen_at=datetime.now(timezone.utc).isoformat(),
        files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
        runtime={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in RUNTIME_FILES},
        model_hashes=model_hashes,versions={name:importlib.metadata.version(name) for name in ('timestral','numpy','torch')})
    with (root/'bundle.json').open('x') as f:json.dump(bundle,f,indent=2)
    for name in [*files,'bundle.json']:(root/name).chmod(0o444)
    print(json.dumps(readiness,indent=2))


def load_bundle(root):
    bundle=json.loads((root/'bundle.json').read_text())
    for name,expected in bundle['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen artifact changed: '+name)
    for name,expected in bundle['runtime'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen runtime changed: '+name)
    for name,expected in bundle['versions'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError('frozen dependency changed')
    config=json.loads((root/'protocol.json').read_text())
    config['bundle_sha256']=hashlib.sha256((root/'bundle.json').read_bytes()).hexdigest()
    return config


def check_clock():
    receipt=public_get('/api/v3/time')
    server=json.loads(receipt['raw_text'])['serverTime']
    mid=(receipt['requested_ms']+receipt['received_ms'])/2
    if abs(server-mid)>2000:raise ValueError('exchange/system clocks differ by over2seconds')
    return receipt


def capture(journal,receipt,label):
    key='source:'+str(receipt['received_ms'])+':'+digest(receipt)[:16]
    journal.append(key,'source',dict(label=label,**receipt))
    return key


def quotes(config,journal,label):
    receipts=[public_get('/api/v3/depth',symbol=s,limit=100) for s in config['symbols']]
    if max(x['received_ms'] for x in receipts)-min(x['requested_ms'] for x in receipts)>10000:
        raise ValueError('cross-asset quote span too wide')
    keys=[capture(journal,r,label) for r in receipts]
    return [json.loads(r['raw_text']) for r in receipts],dict(source_keys=keys,
        earliest_source_requested_ms=min(r['requested_ms'] for r in receipts),
        latest_source_received_ms=max(r['received_ms'] for r in receipts))


def current_wallets(journal):
    wallets={arm:dict(cash=10000.,units=[0.,0.],peak=10000.,halted=False) for arm in ARMS}
    for row in journal.connection.execute("SELECT payload FROM events WHERE kind IN ('entry','mark','final_exit') ORDER BY seq"):
        payload=json.loads(row[0])
        if 'wallets' in payload:wallets=payload['wallets']
    return wallets


def once(root,cache):
    if shutil.disk_usage(root).free<40*1024**3:raise ValueError('less than40GiB free; acquisition stopped')
    config=load_bundle(root)
    journal=Journal(root/'journal.sqlite',config)
    try:
        now=int(time.time()*1000)
        if journal.get('activation') is None:
            clock=check_clock()
            capture(journal,clock,'activation clock')
            journal.append('activation','activation',dict(bundle_sha256=config['bundle_sha256'],
                actual_activation_ms=int(time.time()*1000),first_anchor_ms=config['first_anchor_ms'],
                capital_per_arm=10000,orders=False,classification='paper only'))
        if now>config['last_exit_ms']+600000:
            if journal.get('expired') is None:
                journal.append('expired','expired',dict(actual_ms=now,
                    final_exit_recorded=journal.get('final_exit') is not None,
                    classification='expired; any missing liquidation is retained, not fabricated'))
            return dict(state='expired',**journal.verify())
        history=json.loads((root/'warmup-history.json').read_text())
        for row in journal.connection.execute("SELECT payload FROM events WHERE kind='outcome' ORDER BY seq"):
            history.append(json.loads(row[0])['calibration_row'])
        history.sort(key=lambda x:x['exit_ms'])
        anchors=list(range(config['first_anchor_ms'],config['last_exit_ms']-DAY,WEEK))
        for anchor in anchors:
            decision=journal.get(f'decision:{anchor}')
            if decision and decision['kind']=='forecast' and journal.get(f'ack:{anchor}') and now>=anchor+8*DAY and journal.get(f'outcome:{anchor}') is None:
                receipts=[public_get('/api/v3/klines',symbol=s,interval='1d',startTime=anchor+DAY,
                    endTime=anchor+8*DAY,limit=8) for s in config['symbols']]
                rv=[]
                for receipt in receipts:
                    rows=json.loads(receipt['raw_text'])
                    if [int(x[0]) for x in rows]!=list(range(anchor+DAY,anchor+9*DAY,DAY)):
                        raise ValueError('mature outcome opening prices incomplete')
                    values=[float(x[1]) for x in rows]
                    if any(not math.isfinite(x) or x<=0 for x in values):raise ValueError('invalid outcome price')
                    rv.append(sum(math.log(b/a)**2 for a,b in zip(values,values[1:])))
                keys=[capture(journal,r,'mature weekly variance; last opening price only') for r in receipts]
                raw=decision['payload']['raw_variance']
                label=dict(exit_ms=anchor+8*DAY,timestral=raw['timestral'],ewma=raw['ewma'],realized_variance=rv)
                journal.append(f'outcome:{anchor}','outcome',dict(anchor_ms=anchor,
                    latest_source_received_ms=max(r['received_ms'] for r in receipts),source_keys=keys,calibration_row=label))
                history.append(label)
                history.sort(key=lambda x:x['exit_ms'])
            if anchor<=now<=anchor+config['forecast_deadline_ms'] and decision is None:
                clock=check_clock();capture(journal,clock,'forecast clock')
                receipts=[public_get('/api/v3/klines',symbol=s,interval='1d',startTime=anchor-257*DAY,
                    endTime=anchor-1,limit=257) for s in config['symbols']]
                contexts=[closed_context(json.loads(r['raw_text']),anchor) for r in receipts]
                keys=[capture(journal,r,'causal forecast context') for r in receipts]
                predictor,_=load_model(config,cache)
                median=predictor(contexts)
                result=forecast_weights(contexts,median,history,anchor)
                payload=dict(anchor_ms=anchor,entry_ms=anchor+DAY,exit_ms=anchor+8*DAY,
                    input_end_exclusive_ms=anchor,latest_source_received_ms=max(r['received_ms'] for r in receipts),
                    source_keys=keys,median_log_variances=median,**result)
                if int(time.time()*1000)>anchor+config['forecast_deadline_ms']:
                    event=journal.append(f'decision:{anchor}','missing_forecast',dict(anchor_ms=anchor,
                        reason='source/inference completed after forecast deadline; not retrospectively emitted'))
                else:
                    event=journal.append(f'decision:{anchor}','forecast',payload)
                    if int(time.time()*1000)<=anchor+config['forecast_deadline_ms']:
                        journal.append(f'ack:{anchor}','forecast_ack',dict(anchor_ms=anchor,forecast_hash=event['hash']))
                    else:
                        journal.append(f'unacknowledged:{anchor}','unacknowledged',dict(anchor_ms=anchor,
                            reason='forecast commit returned after ack deadline; ineligible for paper entry'))
                decision=event
            elif now>anchor+config['forecast_deadline_ms'] and decision is None:
                journal.append(f'decision:{anchor}','missing_forecast',dict(anchor_ms=anchor,actual_observation_ms=now,
                    reason='forecast deadline missed; no retrospective reconstruction'))
            entry=anchor+DAY
            if entry<=now<=entry+config['entry_deadline_ms'] and journal.get(f'entry:{anchor}') is None:
                if not decision or decision['kind']!='forecast' or not journal.get(f'ack:{anchor}'):
                    journal.append(f'entry:{anchor}','missing_entry',dict(anchor_ms=anchor,reason='no acknowledged forecast'))
                else:
                    clock=check_clock();capture(journal,clock,'entry clock')
                    books,metadata=quotes(config,journal,'timed paper entry')
                    wallets=current_wallets(journal)
                    fills={}
                    for arm in ARMS:
                        wallet=wallets[arm]
                        liquidation=paper_rebalance(wallet['cash'],wallet['units'],books,[0.,0.])
                        nav=liquidation['cash']
                        wallet['peak']=max(wallet['peak'],nav)
                        if 1-nav/wallet['peak']>=.25:
                            wallet['halted']=True
                        target=[0.,0.] if wallet['halted'] else decision['payload']['weights'][arm]
                        fill=paper_rebalance(wallet['cash'],wallet['units'],books,target)
                        wallet.update(cash=fill['cash'],units=fill['units'])
                        fills[arm]=fill
                    if int(time.time()*1000)>entry+config['entry_deadline_ms']:
                        journal.append(f'entry:{anchor}','missing_entry',dict(anchor_ms=anchor,
                            reason='quotes/calculation finished after entry deadline; holdings unchanged'))
                    else:
                        journal.append(f'entry:{anchor}','entry',dict(anchor_ms=anchor,wallets=wallets,
                            fills=fills,classification='displayed-depth fractional paper scenario, not actual fills',**metadata))
            elif now>entry+config['entry_deadline_ms'] and journal.get(f'entry:{anchor}') is None:
                journal.append(f'entry:{anchor}','missing_entry',dict(anchor_ms=anchor,
                    actual_observation_ms=now,reason='entry deadline missed; holdings unchanged'))
        now=int(time.time()*1000)
        final=config['last_exit_ms']<=now<=config['last_exit_ms']+600000
        mark_key=f'mark:{now//DAY}'
        if (journal.get(mark_key) is None and now>=config['first_anchor_ms']+DAY) or (final and journal.get('final_exit') is None):
            books,metadata=quotes(config,journal,'daily risk mark / final exit' if final else 'daily risk mark')
            wallets=current_wallets(journal)
            marks={}
            for arm in ARMS:
                wallet=wallets[arm]
                liquidation=paper_rebalance(wallet['cash'],wallet['units'],books,[0.,0.])
                nav=liquidation['cash']
                wallet['peak']=max(wallet['peak'],nav)
                dd=1-nav/wallet['peak']
                if dd>=.25 or final:
                    wallet.update(cash=nav,units=[0.,0.],halted=True)
                marks[arm]=dict(liquidation_nav=nav,observed_drawdown=dd,
                    simulated_liquidation=liquidation if dd>=.25 or final else None)
            key,kind=('final_exit','final_exit') if final else (mark_key,'mark')
            journal.append(key,kind,dict(wallets=wallets,marks=marks,**metadata))
        return dict(state='waiting' if now<config['first_anchor_ms'] else 'collecting',
                    first_anchor_ms=config['first_anchor_ms'],last_exit_ms=config['last_exit_ms'],**journal.verify())
    finally:journal.connection.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','once','serve'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--model-cache',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.root,args.model_cache);return
    # This is a bounded local process, not a launchd/cron/system service.
    import fcntl
    with (args.root/'runner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='once':
            print(json.dumps(once(args.root,args.model_cache),indent=2))
            return
        last_state=None
        while True:
            try:
                result=once(args.root,args.model_cache)
            except (urllib.error.URLError,TimeoutError) as exc:
                # Retry only public transport/rate-limit/server failures. The next
                # pass still enforces the original deadlines and records misses.
                if isinstance(exc,urllib.error.HTTPError) and exc.code!=429 and exc.code<500:
                    raise
                print(json.dumps(dict(state='transient_source_failure',type=type(exc).__name__,
                    observed_at=datetime.now(timezone.utc).isoformat(),retry_after_seconds=30)),flush=True)
                time.sleep(30)
                continue
            except Exception as exc:
                # Fail closed; keep the live handle for a supervisor to inspect.
                print(json.dumps(dict(state='error',type=type(exc).__name__,message=str(exc),
                    observed_at=datetime.now(timezone.utc).isoformat())),flush=True)
                raise
            state=(result['state'],result['events'])
            if state!=last_state:
                print(json.dumps(result),flush=True)
                last_state=state
            if result['state']=='expired':break
            time.sleep(5 if result['state']=='collecting' else 30)


if __name__=='__main__':main()
