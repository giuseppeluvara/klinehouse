import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from research.risk_pilot import DAY, Journal, book_value, closed_context, forecast_weights, paper_rebalance


class RiskPilotTests(unittest.TestCase):
    def test_depth_fill_and_no_implicit_borrow(self):
        book={'bids':[['99','1000']], 'asks':[['101','1000']]}
        result=paper_rebalance(10000,[0,0],[book,book],[.5,.5])
        self.assertGreaterEqual(result['cash'],0)
        spending=sum(x['quote_notional']+x['fee'] for x in result['fills'])
        self.assertAlmostEqual(spending,10000,places=6)
        self.assertAlmostEqual(result['mid_nav'],sum(result['units'])*100,places=6)
        with self.assertRaises(ValueError):book_value([['1','2']],3)
        with self.assertRaises(ValueError):
            paper_rebalance(10000,[0,0],[book,book],[.6,.6])

    def test_no_turnover_incurs_no_cost_and_unwind_uses_bid(self):
        book={'bids':[['99','1000']], 'asks':[['101','1000']]}
        result=paper_rebalance(100,[1,1],[book,book],[1/3,1/3])
        self.assertAlmostEqual(result['total_execution_cost'],0,places=8)
        out=paper_rebalance(100,[1,1],[book,book],[0,0])
        self.assertAlmostEqual(out['cash'],100+198*.9998*.999,places=8)
        self.assertEqual(out['units'],[0,0])

    def test_context_rejects_partial_and_future(self):
        anchor=300*DAY
        bars=[[t,'100','102','99','101','1',t+DAY-1,'100',1,'0','0','0']
              for t in range(anchor-257*DAY,anchor,DAY)]
        self.assertEqual(len(closed_context(bars,anchor)),257)
        bars[-1][6]+=1
        with self.assertRaises(ValueError):closed_context(bars,anchor)

    def test_immature_extreme_label_does_not_change_weights(self):
        anchor=400*DAY
        history=[dict(exit_ms=i*7*DAY,timestral=[.001,.002],ewma=[.001,.002],
                      realized_variance=[.001,.002]) for i in range(1,53)]
        closes=[[100*math.exp(i*.001) for i in range(257)]]*2
        medians=[[-10.]*8]*2
        first=forecast_weights(closes,medians,history,anchor)
        history.append(dict(exit_ms=anchor+DAY,timestral=[1.,1.],ewma=[1.,1.],
                            realized_variance=[1e99,1e99]))
        second=forecast_weights(closes,medians,history,anchor)
        self.assertEqual(first,second)
        self.assertEqual(first['calibration_n'],52)

    def test_journal_deadlines_ack_and_immutable_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchor=1000*DAY
            clock=[anchor/1000+1]
            config=dict(first_anchor_ms=anchor,last_exit_ms=anchor+8*DAY,
                        forecast_deadline_ms=600000,entry_deadline_ms=60000)
            journal=Journal(Path(tmp)/'events.sqlite',config,clock=lambda:clock[0])
            payload=dict(anchor_ms=anchor,input_end_exclusive_ms=anchor,entry_ms=anchor+DAY,
                         calibration_max_label_end_ms=anchor-1,latest_source_received_ms=anchor)
            forecast=journal.append(f'decision:{anchor}','forecast',payload)
            clock[0]=(anchor+DAY)/1000+1
            entry=dict(anchor_ms=anchor,latest_source_received_ms=anchor+DAY,
                       earliest_source_requested_ms=anchor+DAY)
            with self.assertRaises(ValueError):journal.append('entry-no-ack','entry',entry)
            with self.assertRaises(ValueError):
                journal.append(f'ack:{anchor}','forecast_ack',dict(anchor_ms=anchor,forecast_hash=forecast['hash']))
            with self.assertRaises(ValueError):journal.append('late','forecast',payload)
            with self.assertRaises(sqlite3.IntegrityError):
                journal.connection.execute("UPDATE events SET kind='forecast_ack'")
            self.assertEqual(journal.verify()['events'],1)
            journal.connection.close()

    def test_valid_ack_entry_and_outcome_maturity(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchor=1000*DAY
            clock=[anchor/1000+1]
            config=dict(first_anchor_ms=anchor,last_exit_ms=anchor+8*DAY,
                        forecast_deadline_ms=600000,entry_deadline_ms=60000)
            journal=Journal(Path(tmp)/'events.sqlite',config,clock=lambda:clock[0])
            payload=dict(anchor_ms=anchor,input_end_exclusive_ms=anchor,entry_ms=anchor+DAY,
                         calibration_max_label_end_ms=anchor-1,latest_source_received_ms=anchor)
            f=journal.append(f'decision:{anchor}','forecast',payload)
            journal.append(f'ack:{anchor}','forecast_ack',dict(anchor_ms=anchor,forecast_hash=f['hash']))
            with self.assertRaises(ValueError):
                journal.append('early','outcome',dict(anchor_ms=anchor,latest_source_received_ms=anchor))
            clock[0]=(anchor+DAY)/1000+1
            journal.append('entry','entry',dict(anchor_ms=anchor,latest_source_received_ms=anchor+DAY,
                           earliest_source_requested_ms=anchor+DAY))
            self.assertEqual(journal.verify()['events'],3)
            journal.connection.close()


if __name__=='__main__':unittest.main()
