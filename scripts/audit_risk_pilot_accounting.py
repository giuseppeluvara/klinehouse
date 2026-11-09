"""Independent ledger accounting; imports no production portfolio/quote engine."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math
import urllib.parse


class AccountingAuditError(ValueError):
    pass


def need(condition,message):
    if not condition:raise AccountingAuditError(message)


def number(value):
    try:result=Decimal(str(value))
    except Exception as exc:raise AccountingAuditError('invalid decimal') from exc
    need(result.is_finite(),'nonfinite accounting value')
    return result


def near(actual,expected,label,tolerance='0.000001'):
    need(abs(number(actual)-number(expected))<=Decimal(tolerance),'accounting mismatch: '+label)


def consume(levels,quantity):
    remaining=number(quantity)
    need(remaining>=0,'negative quantity')
    result=Decimal(0)
    for level in levels:
        need(len(level)==2,'invalid depth level')
        price,size=map(number,level)
        need(price>0 and size>0,'nonpositive depth')
        used=min(size,remaining)
        result+=used*price
        remaining-=used
        if remaining<=Decimal('1e-12'):return result
    need(remaining<=Decimal('1e-12'),'insufficient recorded depth')
    return result


def books_for(event,by_key,symbols):
    keys=event['payload']['source_keys']
    need(len(keys)==len(symbols),'wrong number of quote sources')
    books=[]
    for key,symbol in zip(keys,symbols):
        need(key in by_key,'missing quote source')
        source=by_key[key]
        receipt=source['payload']
        need(source['kind']=='source' and source['seq']<event['seq'],'quote source not committed before portfolio event')
        need(receipt['requested_ms']<=receipt['received_ms']<=source['recorded_ms']<=event['recorded_ms'],
             'invalid quote receipt chronology')
        need(hashlib.sha256(receipt['raw_text'].encode()).hexdigest()==receipt['sha256'],'raw quote hash mismatch')
        parsed=urllib.parse.urlparse(receipt['url'])
        need(parsed.scheme=='https' and parsed.netloc=='api.binance.com' and parsed.path=='/api/v3/depth',
             'nonpublic or wrong quote endpoint')
        need(urllib.parse.parse_qs(parsed.query).get('symbol')==[symbol],'quote assigned to wrong asset')
        book=json.loads(receipt['raw_text'])
        for key,reverse in (('bids',True),('asks',False)):
            need(bool(book[key]),'empty depth')
            prices=[number(x[0]) for x in book[key]]
            need(prices==sorted(set(prices),reverse=reverse),'unordered or duplicate depth')
            for level in book[key]:consume([level],Decimal(0))
        need(number(book['bids'][0][0])<number(book['asks'][0][0]),'crossed depth')
        books.append(book)
    return books


def audit_accounting(events,config):
    """Reconstruct cash, units, costs, weights and risk halts using Decimal.

    This validates recorded paper scenarios, not actual fills or profitability.
    It independently derives transactions rather than solving target portfolios.
    """
    with localcontext() as context:
        context.prec=40
        arms=config.get('arms',['timestral','ewma','fixed'])
        symbols=config['symbols']
        need(len(symbols)==2,'two symbols required')
        initial=number(config.get('initial_paper_capital_per_arm',10000))
        need(initial>0,'nonpositive initial capital')
        fee,reserve=Decimal('.001'),Decimal('.0002')
        wallets={arm:dict(cash=initial,units=[Decimal(0),Decimal(0)],peak=initial,halted=False) for arm in arms}
        cumulative_fees={arm:Decimal(0) for arm in arms}
        latest_nav={arm:initial for arm in arms}
        by_key={e['event_key']:e for e in events}
        checked_entries=checked_marks=checked_fills=0
        first_week=None
        anchor=config.get('first_anchor_ms')
        if anchor is not None:
            day=86_400_000
            boundary=anchor+8*day
            first_entry=by_key.get(f'entry:{anchor}')
            outcome=by_key.get(f'outcome:{anchor}')
            first_week={'status':'awaiting_outcome','boundary_ms':boundary}
            if first_entry and first_entry['kind']!='entry':
                first_week['status']='missing_first_entry'
            elif outcome:
                first_week['status']='awaiting_boundary_quotes' if first_entry else 'missing_first_entry'
            week_drawdown={arm:Decimal(0) for arm in arms}
            mark_days=set()

        def liquidation(wallet,books):
            return wallet['cash']+sum(consume(book['bids'],q)*(1-reserve)*(1-fee)
                                     for book,q in zip(books,wallet['units']))

        def verify_fills(fill,wallet,books,expected_units):
            nonlocal checked_fills
            need(len(fill['fills'])==2,'wrong number of fills')
            cash=wallet['cash']
            fees=Decimal(0)
            for old,new,book,record in zip(wallet['units'],expected_units,books,fill['fills']):
                delta=new-old
                side='buy' if delta>=0 else 'sell'
                need(record['side']==side,'wrong fill side')
                near(record['quantity'],abs(delta),'fill quantity','0.00000001')
                value=consume(book['asks' if side=='buy' else 'bids'],abs(delta))
                value*=1+reserve if side=='buy' else 1-reserve
                commission=value*fee
                near(record['quote_notional'],value,'fill notional')
                near(record['fee'],commission,'fill fee')
                cash+=-value-commission if side=='buy' else value-commission
                fees+=commission
                checked_fills+=1
            near(fill['cash'],cash,'fill cash')
            need(cash>=Decimal('-0.000001'),'implicit borrowing')
            near(fill['explicit_fee'],fees,'total fee')
            for actual,expected in zip(fill['units'],expected_units):near(actual,expected,'fill units','0.00000001')
            return cash,fees

        for event in events:
            kind=event['kind']
            if kind not in ('entry','mark','final_exit'):continue
            payload=event['payload']
            need(set(payload['wallets'])==set(arms),'missing or unregistered portfolio arm')
            books=books_for(event,by_key,symbols)
            mids=[(number(b['bids'][0][0])+number(b['asks'][0][0]))/2 for b in books]
            first_week_boundary=False
            in_first_week=False
            if first_week and first_entry and first_entry['kind']=='entry' and 'arms' not in first_week:
                receipts=[by_key[key]['payload'] for key in payload['source_keys']]
                request_ms=min(r['requested_ms'] for r in receipts)
                quote_ms=max(r['received_ms'] for r in receipts)
                in_first_week=first_entry['seq']<=event['seq'] and quote_ms<boundary
                if in_first_week and kind=='mark':
                    mark_days.add(quote_ms//day)
                first_week_boundary=(outcome is not None and boundary<=request_ms<=quote_ms<boundary+day
                    and (kind=='mark' or (kind=='entry' and payload['anchor_ms']==anchor+7*day)))
                if first_week_boundary:
                    first_week.update(status='valued',quote_ms=quote_ms,boundary_event=event['event_key'],
                                      daily_marks=len(mark_days),expected_daily_marks=7,arms={})
            for arm in arms:
                old=wallets[arm]
                recorded=payload['wallets'][arm]
                liq=liquidation(old,books)
                peak=max(old['peak'],liq)
                dd=1-liq/peak
                halted=old['halted'] or dd>=Decimal('.25') or kind=='final_exit'
                if in_first_week or first_week_boundary:
                    week_drawdown[arm]=max(week_drawdown[arm],dd)
                if first_week_boundary:
                    pnl=liq-initial
                    first_week['arms'][arm]=dict(net_pnl_usdt=float(pnl),net_return_pct=float(100*pnl/initial),
                        liquidation_nav_usdt=float(liq),observed_max_drawdown_pct=float(100*week_drawdown[arm]),
                        risk_halted=halted)
                if kind=='entry':
                    need(f"decision:{payload['anchor_ms']}" in by_key,'entry without decision')
                    decision=by_key[f"decision:{payload['anchor_ms']}"]
                    need(decision['kind']=='forecast' and decision['seq']<event['seq'],'entry decision not a prior forecast')
                    targets=[Decimal(0),Decimal(0)] if halted else list(map(number,decision['payload']['weights'][arm]))
                    need(len(targets)==2 and min(targets)>=0 and sum(targets)<=1,'invalid target weights')
                    new_units=list(map(number,recorded['units']))
                    need(len(new_units)==2 and min(new_units)>=0,'invalid recorded units')
                    fill=payload['fills'][arm]
                    cash,fees=verify_fills(fill,old,books,new_units)
                    cumulative_fees[arm]+=fees
                    after_mid=cash+sum(q*p for q,p in zip(new_units,mids))
                    before_mid=old['cash']+sum(q*p for q,p in zip(old['units'],mids))
                    near(fill['mid_nav'],after_mid,'post-cost mid NAV')
                    near(fill['total_execution_cost'],before_mid-after_mid,'total execution cost')
                    for q,p,w in zip(new_units,mids,targets):near(q*p,w*after_mid,'target allocation')
                else:
                    mark=payload['marks'][arm]
                    near(mark['liquidation_nav'],liq,'liquidation NAV')
                    near(mark['observed_drawdown'],dd,'observed drawdown','0.000000001')
                    should_close=dd>=Decimal('.25') or kind=='final_exit'
                    if should_close:
                        need(mark['simulated_liquidation'] is not None,'missing risk/final liquidation')
                        cash,fees=verify_fills(mark['simulated_liquidation'],old,books,[Decimal(0),Decimal(0)])
                        cumulative_fees[arm]+=fees
                        new_units=[Decimal(0),Decimal(0)]
                    else:
                        need(mark['simulated_liquidation'] is None,'unexpected liquidation')
                        cash,new_units=old['cash'],old['units']
                near(recorded['cash'],cash,'wallet cash')
                for actual,expected in zip(recorded['units'],new_units):near(actual,expected,'wallet units','0.00000001')
                near(recorded['peak'],peak,'risk high-water mark')
                need(recorded['halted'] is halted,'risk halt mismatch')
                wallets[arm]=dict(cash=cash,units=new_units,peak=peak,halted=halted)
                latest_nav[arm]=liquidation(wallets[arm],books)
            if kind=='entry':checked_entries+=1
            else:checked_marks+=1
        result=dict(status='passed',paper_entries_checked=checked_entries,marks_or_final_exits_checked=checked_marks,
            individual_paper_fills_checked=checked_fills,
            arms={arm:dict(cash=float(w['cash']),units=list(map(float,w['units'])),halted=w['halted'],
                cumulative_simulated_fees=float(cumulative_fees[arm]),
                latest_derived_liquidation_nav=float(latest_nav[arm])) for arm,w in wallets.items()},
            interpretation='independent paper-accounting reconstruction; no actual fills or prospective profit certification')
        if first_week is not None:
            if first_week['status']=='valued' and {'timestral','ewma'}<=set(arms):
                first_week['timestral_minus_ewma_pp']=(first_week['arms']['timestral']['net_return_pct']
                    -first_week['arms']['ewma']['net_return_pct'])
            result['first_week']=first_week
        return result
