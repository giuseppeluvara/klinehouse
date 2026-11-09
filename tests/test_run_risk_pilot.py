from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.risk_pilot import DAY, Journal
from scripts.run_risk_pilot import once


class RiskPilotLifecycleTests(unittest.TestCase):
    def test_committed_forecast_entry_maturity_and_risk_halt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            anchor=int(datetime(2026,11,9,tzinfo=timezone.utc).timestamp()*1000)
            config=dict(first_anchor_ms=anchor,last_exit_ms=anchor+15*DAY,
                        forecast_deadline_ms=600000,entry_deadline_ms=60000,
                        bundle_sha256='a'*64,symbols=['BTCUSDT','ETHUSDT'])
            history=[dict(exit_ms=anchor-(52-i)*7*DAY,timestral=[.0001,.0001],
                          ewma=[.0001,.0001],realized_variance=[.0001,.0001])
                     for i in range(52)]
            (root/'warmup-history.json').write_text(json.dumps(history))
            state={'ms':anchor+1000,'quote':130.}

            def api(path,**params):
                requested=state['ms']
                state['ms']+=10
                if path.endswith('/time'):
                    body={'serverTime':state['ms']}
                elif path.endswith('/depth'):
                    p=state['quote']
                    body={'bids':[[str(p-.01),'10000']], 'asks':[[str(p+.01),'10000']]}
                else:
                    start=params['startTime'];limit=params['limit']
                    body=[]
                    for i in range(limit):
                        t=start+i*DAY
                        close=100+(t-(anchor-257*DAY))/DAY*.1
                        body.append([t,str(close-.01),str(close+1),str(close-1),str(close),
                            '1000',t+DAY-1,'100000',100,'500','50000','0'])
                return dict(url=path+json.dumps(params,sort_keys=True),requested_ms=requested,
                    received_ms=state['ms'],raw_text=json.dumps(body),sha256='a'*64)

            with patch('scripts.run_risk_pilot.load_bundle',return_value=config), \
                 patch('scripts.run_risk_pilot.public_get',side_effect=api), \
                 patch('scripts.run_risk_pilot.load_model',return_value=(lambda _: [[-10.]*8]*2,{})), \
                 patch('scripts.run_risk_pilot.shutil.disk_usage',return_value=SimpleNamespace(free=100*1024**3)), \
                 patch('scripts.run_risk_pilot.time.time',side_effect=lambda:state['ms']/1000):
                once(root,Path(tmp))
                journal=Journal(root/'journal.sqlite',config)
                self.assertEqual(journal.get(f'decision:{anchor}')['kind'],'forecast')
                self.assertIsNotNone(journal.get(f'ack:{anchor}'))
                self.assertIsNone(journal.get(f'entry:{anchor}'))
                journal.connection.close()
                state['ms']=anchor+DAY+1000
                once(root,Path(tmp))
                journal=Journal(root/'journal.sqlite',config)
                entry=journal.get(f'entry:{anchor}')
                self.assertEqual(entry['kind'],'entry')
                for wallet in entry['payload']['wallets'].values():
                    self.assertGreaterEqual(wallet['cash'],0)
                    self.assertGreater(sum(wallet['units']),0)
                self.assertIsNone(journal.get(f'outcome:{anchor}'))
                journal.connection.close()
                # A violent observed gap must be recorded at its actual quotes,
                # rather than inventing a fill at a guaranteed25% stop price.
                state['ms']=anchor+2*DAY+1000
                state['quote']=1.
                once(root,Path(tmp))
                journal=Journal(root/'journal.sqlite',config)
                mark=journal.get(f"mark:{state['ms']//DAY}")
                self.assertTrue(mark['payload']['wallets']['fixed']['halted'])
                self.assertGreater(mark['payload']['marks']['fixed']['observed_drawdown'],.25)
                journal.connection.close()
                state['ms']=anchor+8*DAY+1000
                once(root,Path(tmp))
                journal=Journal(root/'journal.sqlite',config)
                outcome=journal.get(f'outcome:{anchor}')
                self.assertEqual(outcome['payload']['calibration_row']['exit_ms'],anchor+8*DAY)
                # Jumping past the next Monday records the missed slot honestly.
                self.assertEqual(journal.get(f'decision:{anchor+7*DAY}')['kind'],'missing_forecast')
                self.assertEqual(journal.get(f'entry:{anchor+7*DAY}')['kind'],'missing_entry')
                self.assertTrue(journal.verify()['events']>10)
                journal.connection.close()


if __name__=='__main__':unittest.main()
