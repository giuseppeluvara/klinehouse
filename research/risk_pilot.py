"""Causal decisions and durable evidence for an explicitly paper-only risk trial.

No exchange/account client or environment credentials are used by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

DAY = 86400000
WEEK = 7 * DAY
ARMS = ('timestral', 'ewma', 'fixed')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def forecast_weights(closes, median_log_variances, history, anchor_ms):
    """Same 126-day trend / calibrated risk policy as the historical replay.

    ``closes`` contains exactly 257 already closed UTC days per asset. History
    uses label-end timestamps, not the date a forecast was generated.
    """
    if len(closes) != 2 or any(len(x) != 257 for x in closes):
        raise ValueError('257 closed days for each of two assets required')
    if len(median_log_variances) != 2 or any(len(x) != 8 for x in median_log_variances):
        raise ValueError('eight forecast medians for each asset required')
    if any(not math.isfinite(p) or p <= 0 for asset in closes for p in asset):
        raise ValueError('invalid historical close')
    if any(not math.isfinite(p) for asset in median_log_variances for p in asset):
        raise ValueError('invalid model forecast')
    ends = [x['exit_ms'] for x in history]
    if ends != sorted(set(ends)):
        raise ValueError('duplicate or unordered calibration labels')
    mature = [x for x in history if x['exit_ms'] <= anchor_ms][-52:]
    if len(mature) < 26:
        raise ValueError('insufficient matured calibration')
    for row in mature:
        for key in ('timestral', 'ewma', 'realized_variance'):
            if len(row[key]) != 2 or any(not math.isfinite(x) or x < 0 for x in row[key]):
                raise ValueError('invalid calibration row')
            if key != 'realized_variance' and min(row[key]) <= 0:
                raise ValueError('nonpositive historical forecast')
    decay = [math.exp(-math.log(2)*(255-i)/20) for i in range(256)]
    raw = {'timestral': [], 'ewma': []}
    scales = {'timestral': [], 'ewma': []}
    weights = {arm: [] for arm in ARMS}
    trend = []
    for asset in range(2):
        r2 = [math.log(b/a)**2 for a,b in zip(closes[asset], closes[asset][1:])]
        raw['timestral'].append(sum(math.exp(x) for x in median_log_variances[asset][1:8]))
        raw['ewma'].append(7*sum(w*r for w,r in zip(decay,r2))/sum(decay))
        up = math.log(closes[asset][-1]/closes[asset][-127]) > 0
        trend.append(up)
        for arm in ('timestral', 'ewma'):
            scale = sum(x['realized_variance'][asset]/x[arm][asset] for x in mature)/len(mature)
            variance = raw[arm][asset]*scale
            if not math.isfinite(variance) or variance <= 0:
                raise ValueError('nonpositive calibrated risk')
            scales[arm].append(scale)
            weights[arm].append(min(.5, .1/math.sqrt(variance*365/7))*up)
        weights['fixed'].append(.5*up)
    return dict(weights=weights, trend=trend, raw_variance=raw, calibration=scales,
                calibration_n=len(mature), calibration_max_label_end_ms=mature[-1]['exit_ms'])


def closed_context(bars, anchor_ms):
    """Reject gaps, partial bars, OHLC inconsistencies and future input."""
    expected = list(range(anchor_ms-257*DAY, anchor_ms, DAY))
    if len(bars) != 257 or [int(x[0]) for x in bars] != expected:
        raise ValueError('context must contain exactly 257 consecutive closed UTC days')
    for row in bars:
        if len(row) != 12 or int(row[6]) != int(row[0])+DAY-1 or int(row[6]) >= anchor_ms:
            raise ValueError('partial or future candle')
        o,h,l,c = (float(row[k]) for k in (1,2,3,4))
        if not all(math.isfinite(x) and x > 0 for x in (o,h,l,c)) or not l <= min(o,c) <= max(o,c) <= h:
            raise ValueError('inconsistent OHLC')
    return [float(x[4]) for x in bars]


def book_value(levels, quantity):
    """Consume displayed levels; insufficient depth invalidates the paper fill."""
    if not math.isfinite(quantity) or quantity < 0:
        raise ValueError('invalid quantity')
    remaining, value = quantity, 0.0
    for price, size in levels:
        p,q = float(price), float(size)
        if not math.isfinite(p) or not math.isfinite(q) or p <= 0 or q <= 0:
            raise ValueError('invalid depth level')
        amount = min(remaining, q)
        value += amount*p
        remaining -= amount
        if remaining <= 1e-12:
            return value
    if remaining > 1e-12:
        raise ValueError('insufficient displayed depth')
    return value


def validate_book(book):
    bids, asks = book['bids'], book['asks']
    if not bids or not asks:
        raise ValueError('empty book')
    for side, descending in ((bids,True),(asks,False)):
        prices = [float(x[0]) for x in side]
        if prices != sorted(set(prices), reverse=descending):
            raise ValueError('unordered or duplicate depth')
        for level in side:
            book_value([level], min(float(level[1]), 1.0))
    if float(bids[0][0]) >= float(asks[0][0]):
        raise ValueError('crossed book')


def paper_rebalance(cash, units, books, weights, fee=.001, reserve=.0002):
    """Fractional-quantity paper replay of depth, never actual exchange fills.

    Solve for post-cost mid NAV with buy and sell VWAP from displayed books.
    There is no assumption of simultaneous live executability across assets.
    """
    if len(units) != 2 or len(books) != 2 or len(weights) != 2:
        raise ValueError('two assets required')
    if any(not math.isfinite(x) or x < 0 for x in [cash,*units,*weights]):
        raise ValueError('invalid cash, units or weights')
    if sum(weights) > 1+1e-12 or not 0 <= fee < 1 or not 0 <= reserve < 1:
        raise ValueError('leverage or invalid costs')
    for book in books:
        validate_book(book)
    mids = [(float(b['bids'][0][0])+float(b['asks'][0][0]))/2 for b in books]
    nav = cash+sum(u*p for u,p in zip(units,mids))

    def trial(value):
        target = [w*value/p for w,p in zip(weights,mids)]
        balance = cash
        fills = []
        for old,new,book in zip(units,target,books):
            delta = new-old
            if delta >= 0:
                notional = book_value(book['asks'],delta)*(1+reserve)
                amount = notional*(1+fee)
                balance -= amount
                fills.append(dict(side='buy',quantity=delta,quote_notional=notional,fee=notional*fee))
            else:
                notional = book_value(book['bids'],-delta)*(1-reserve)
                amount = notional*(1-fee)
                balance += amount
                fills.append(dict(side='sell',quantity=-delta,quote_notional=notional,fee=notional*fee))
        return balance,target,fills

    low,high = 0.0,nav
    for _ in range(70):
        value = (low+high)/2
        balance,_,_ = trial(value)
        if balance < (1-sum(weights))*value:
            high = value
        else:
            low = value
    value = (low+high)/2
    balance,target,fills = trial(value)
    if balance < -1e-7 or abs(balance-(1-sum(weights))*value) > max(1e-7,nav*1e-10):
        raise ValueError('paper portfolio is not self financing')
    return dict(cash=max(0.0,balance),units=target,fills=fills,
                mid_nav=value,explicit_fee=sum(f['fee'] for f in fills),
                total_execution_cost=nav-value)


class Journal:
    """Append-only transactional evidence. Hashes do not attest an entire DB rewrite."""
    def __init__(self,path,config,clock=None):
        self.clock=clock or time.time
        self.config=config
        self.connection=sqlite3.connect(path,timeout=10)
        self.connection.row_factory=sqlite3.Row
        self.connection.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,event_key TEXT UNIQUE NOT NULL,
                kind TEXT NOT NULL,recorded_ms INTEGER NOT NULL,payload TEXT NOT NULL,
                previous TEXT NOT NULL,hash TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS no_event_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TRIGGER IF NOT EXISTS no_event_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT,'append only'); END;
            CREATE TRIGGER IF NOT EXISTS no_meta_update BEFORE UPDATE ON metadata
                BEGIN SELECT RAISE(ABORT,'immutable metadata'); END;
            CREATE TRIGGER IF NOT EXISTS no_meta_delete BEFORE DELETE ON metadata
                BEGIN SELECT RAISE(ABORT,'immutable metadata'); END;
        ''')
        encoded=canonical(config)
        found=self.connection.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
        if found is None:
            with self.connection:self.connection.execute("INSERT INTO metadata VALUES('config',?)",(encoded,))
        elif found[0]!=encoded:
            raise ValueError('frozen configuration differs')
        self.verify()

    def get(self,key):
        row=self.connection.execute('SELECT * FROM events WHERE event_key=?',(key,)).fetchone()
        if row is None:return None
        return dict(seq=row['seq'],kind=row['kind'],recorded_ms=row['recorded_ms'],
                    payload=json.loads(row['payload']),hash=row['hash'])

    def append(self,key,kind,payload):
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            now=int(self.clock()*1000)
            anchor=payload.get('anchor_ms')
            if kind=='forecast':
                if anchor is None or not self.config['first_anchor_ms'] <= anchor < self.config['last_exit_ms']-DAY:
                    raise ValueError('forecast outside trial')
                if (anchor-self.config['first_anchor_ms'])%WEEK:
                    raise ValueError('unregistered forecast anchor')
                if not anchor <= now <= anchor+self.config['forecast_deadline_ms']:
                    raise ValueError('late or backdated forecast')
                if payload['input_end_exclusive_ms']!=anchor or payload['entry_ms']!=anchor+DAY:
                    raise ValueError('noncausal forecast interval')
                if payload['calibration_max_label_end_ms']>anchor:
                    raise ValueError('immature calibration')
                if payload['latest_source_received_ms']>now:
                    raise ValueError('future receipt')
            if kind=='forecast_ack':
                forecast=self.get(f'decision:{anchor}')
                if forecast is None or forecast['kind']!='forecast' or forecast['hash']!=payload['forecast_hash']:
                    raise ValueError('ack has no matching forecast')
                if now<forecast['recorded_ms'] or now>anchor+self.config['forecast_deadline_ms']:
                    raise ValueError('late forecast acknowledgement')
            if kind=='entry':
                forecast=self.get(f'decision:{anchor}')
                ack=self.get(f'ack:{anchor}')
                if not forecast or forecast['kind']!='forecast' or not ack:
                    raise ValueError('entry requires a durable forecast and ack')
                entry=anchor+DAY
                if not entry<=now<=entry+self.config['entry_deadline_ms']:
                    raise ValueError('late or backdated paper entry')
                if payload['latest_source_received_ms']>now or payload['earliest_source_requested_ms']<entry:
                    raise ValueError('paper entry quotes outside entry interval')
            if kind=='outcome':
                forecast=self.get(f'decision:{anchor}')
                if not forecast or forecast['kind']!='forecast' or now<anchor+8*DAY:
                    raise ValueError('outcome has not matured')
                if payload['latest_source_received_ms']>now:
                    raise ValueError('future outcome receipt')
            tail=self.connection.execute('SELECT seq,hash,recorded_ms FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            if tail and now<tail['recorded_ms']:
                raise ValueError('clock moved backwards')
            seq=tail['seq']+1 if tail else 1
            previous=tail['hash'] if tail else '0'*64
            event=dict(seq=seq,event_key=key,kind=kind,recorded_ms=now,payload=payload,previous=previous)
            value=digest(event)
            self.connection.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?)',
                (seq,key,kind,now,canonical(payload),previous,value))
        return self.get(key)

    def verify(self):
        previous='0'*64
        count=0
        for row in self.connection.execute('SELECT * FROM events ORDER BY seq'):
            count+=1
            event=dict(seq=row['seq'],event_key=row['event_key'],kind=row['kind'],
                recorded_ms=row['recorded_ms'],payload=json.loads(row['payload']),previous=row['previous'])
            if row['seq']!=count or row['previous']!=previous or digest(event)!=row['hash']:
                raise ValueError('journal hash chain mismatch')
            previous=row['hash']
        return dict(events=count,head=previous)
