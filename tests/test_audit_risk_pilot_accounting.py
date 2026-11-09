import copy
import hashlib
import json
import unittest

from research.risk_pilot import DAY, paper_rebalance
from scripts.audit_risk_pilot_accounting import AccountingAuditError, audit_accounting


def fixture():
    anchor=1794182400000
    config=dict(symbols=['BTCUSDT','ETHUSDT'],arms=['timestral','ewma','fixed'],
                initial_paper_capital_per_arm=10000)
    targets=dict(timestral=[.4,.3],ewma=[.2,.2],fixed=[.5,.5])
    books=[dict(bids=[['99','10'],['98','500']],asks=[['101','10'],['102','500']]),
           dict(bids=[['49','20'],['48','500']],asks=[['51','20'],['52','500']])]
    events=[]

    def event(key,kind,payload,stamp):
        events.append(dict(seq=len(events)+1,event_key=key,kind=kind,
                           recorded_ms=stamp,payload=copy.deepcopy(payload)))

    def sources(books,stamp,label):
        keys=[]
        for i,(book,symbol) in enumerate(zip(books,config['symbols'])):
            key=f'source:{label}:{symbol}'
            raw=json.dumps(book)
            event(key,'source',dict(url='https://api.binance.com/api/v3/depth?symbol='+symbol,
                  raw_text=raw,sha256=hashlib.sha256(raw.encode()).hexdigest(),
                  requested_ms=stamp-5+i,received_ms=stamp-3+i),stamp+i)
            keys.append(key)
        return keys

    event(f'decision:{anchor}','forecast',dict(weights=targets),anchor+1000)
    keys=sources(books,anchor+DAY+1000,'entry')
    wallets={};fills={}
    for arm in config['arms']:
        f=paper_rebalance(10000,[0,0],books,targets[arm])
        fills[arm]=f
        wallets[arm]=dict(cash=f['cash'],units=f['units'],peak=10000.,halted=False)
    event('entry','entry',dict(anchor_ms=anchor,source_keys=keys,wallets=wallets,fills=fills),anchor+DAY+1010)
    gap_books=[dict(bids=[['9.99','1000']],asks=[['10.01','1000']]),
               dict(bids=[['4.99','1000']],asks=[['5.01','1000']])]
    for label,kind,stamp in [('gap','mark',anchor+2*DAY),('final','final_exit',anchor+8*DAY)]:
        keys=sources(gap_books,stamp+1000,label)
        marks={}
        for arm in config['arms']:
            old=wallets[arm]
            liquidation=paper_rebalance(old['cash'],old['units'],gap_books,[0,0])
            nav=liquidation['cash'];peak=max(old['peak'],nav);dd=1-nav/peak
            marks[arm]=dict(liquidation_nav=nav,observed_drawdown=dd,simulated_liquidation=liquidation)
            wallets[arm]=dict(cash=nav,units=[0.,0.],peak=peak,halted=True)
        event(label,kind,dict(source_keys=keys,wallets=wallets,marks=marks),stamp+1010)
    return events,config,books


class AccountingAuditTests(unittest.TestCase):
    def test_first_week_uses_pre_rebalance_holdings(self):
        events,config,books=fixture()
        opening=next(e for e in events if e['kind']=='entry')
        anchor=opening['payload']['anchor_ms']
        config['first_anchor_ms']=anchor
        opening['event_key']=f'entry:{anchor}'
        self.assertEqual(audit_accounting(events,config)['first_week']['status'],'awaiting_outcome')

        boundary=anchor+8*DAY
        events=[e for e in events if e['event_key']!='gap' and ':gap:' not in e['event_key']]
        closing=events.pop()
        closing.update(kind='entry',event_key=f'entry:{anchor+7*DAY}')
        targets={arm:[.5,.5] for arm in config['arms']}
        events.append(dict(seq=closing['seq'],kind='forecast',event_key=f'decision:{anchor+7*DAY}',
            recorded_ms=boundary-DAY,payload={'weights':targets}))
        closing['seq']+=1
        closing['payload'].update(anchor_ms=anchor+7*DAY,fills={})
        for source,book in zip((e for e in events if ':final:' in e['event_key']),books):
            raw=json.dumps(book)
            source['payload'].update(raw_text=raw,sha256=hashlib.sha256(raw.encode()).hexdigest())
        for arm,old in opening['payload']['wallets'].items():
            fill=paper_rebalance(old['cash'],old['units'],books,targets[arm])
            closing['payload']['wallets'][arm]=dict(cash=fill['cash'],units=fill['units'],peak=10000.,halted=False)
            closing['payload']['fills'][arm]=fill
        events.extend([closing,dict(seq=closing['seq']+1,kind='outcome',event_key=f'outcome:{anchor}',
            recorded_ms=boundary+2000,payload={})])

        result=audit_accounting(events,config)
        week=result['first_week']
        self.assertEqual(week['status'],'valued')
        self.assertEqual(week['boundary_event'],closing['event_key'])
        old=opening['payload']['wallets']['timestral']
        expected=paper_rebalance(old['cash'],old['units'],books,[0,0])['cash']
        self.assertAlmostEqual(week['arms']['timestral']['net_pnl_usdt'],expected-10000)
        self.assertAlmostEqual(week['arms']['timestral']['observed_max_drawdown_pct'],(1-expected/10000)*100)
        self.assertNotAlmostEqual(result['arms']['timestral']['latest_derived_liquidation_nav'],expected)

    def test_reconstructs_multilevel_fills_gap_and_final_exit(self):
        events,config,_=fixture()
        result=audit_accounting(events,config)
        self.assertEqual(result['paper_entries_checked'],1)
        self.assertEqual(result['marks_or_final_exits_checked'],2)
        self.assertEqual(result['individual_paper_fills_checked'],18)
        self.assertTrue(all(w['halted'] for w in result['arms'].values()))
        self.assertTrue(all(w['cumulative_simulated_fees']>0 for w in result['arms'].values()))

    def test_rejects_consistent_accounting_with_wrong_strategy_weight(self):
        events,config,books=fixture()
        entry=next(x['payload'] for x in events if x['kind']=='entry')
        wrong=paper_rebalance(10000,[0,0],books,[.3,.3])
        entry['fills']['timestral']=wrong
        entry['wallets']['timestral'].update(cash=wrong['cash'],units=wrong['units'])
        with self.assertRaises(AccountingAuditError):audit_accounting(events,config)

    def test_rejects_fee_or_cash_fabrication(self):
        for field in ('fee','cash'):
            events,config,_=fixture()
            entry=next(x['payload'] for x in events if x['kind']=='entry')
            if field=='fee':entry['fills']['timestral']['fills'][0]['fee']+=1
            else:entry['wallets']['timestral']['cash']+=1
            with self.assertRaises(AccountingAuditError):audit_accounting(events,config)

    def test_rejects_unreported_risk_halt(self):
        events,config,_=fixture()
        next(x['payload'] for x in events if x['kind']=='mark')['wallets']['fixed']['halted']=False
        with self.assertRaises(AccountingAuditError):audit_accounting(events,config)

    def test_rejects_insufficient_depth_even_with_valid_body_hash(self):
        events,config,_=fixture()
        source=next(x['payload'] for x in events if x['event_key']=='source:entry:BTCUSDT')
        book=json.loads(source['raw_text']);book['asks'][1][1]='.000001'
        source['raw_text']=json.dumps(book)
        source['sha256']=hashlib.sha256(source['raw_text'].encode()).hexdigest()
        with self.assertRaises(AccountingAuditError):audit_accounting(events,config)

    def test_rejects_asset_swap_and_nonfinite_balance(self):
        for corruption in ('symbol','nan'):
            events,config,_=fixture()
            if corruption=='symbol':
                source=next(x['payload'] for x in events if x['event_key']=='source:entry:BTCUSDT')
                source['url']=source['url'].replace('BTCUSDT','ETHUSDT')
            else:
                entry=next(x['payload'] for x in events if x['kind']=='entry')
                entry['wallets']['timestral']['cash']=float('nan')
            with self.assertRaises(AccountingAuditError):audit_accounting(events,config)


if __name__=='__main__':unittest.main()
