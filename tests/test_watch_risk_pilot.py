import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import watch_risk_pilot as monitor


class LocalAlertTests(unittest.TestCase):
    def setUp(self):
        self.anchor = 1794182400000
        self.entry = self.anchor + monitor.DAY
        self.config = dict(first_anchor_ms=self.anchor, last_exit_ms=self.anchor + 8 * monitor.DAY,
                           entry_deadline_ms=60000, maximum_observed_drawdown=0.25)

    def test_late_absence_and_recorded_miss_share_one_alert_across_restarts(self):
        deadline = self.entry + self.config['entry_deadline_ms']
        key = f'entry:{self.anchor}'
        self.assertEqual(list(monitor.missed_entries(self.config, {}, deadline)), [])
        self.assertEqual(list(monitor.missed_entries(self.config, {key: ('entry', {})}, deadline + 20000)), [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'protocol.json').write_text(json.dumps(self.config))
            with patch.object(monitor.time, 'time_ns', return_value=(deadline + 20000) * 1000000), \
                 patch.object(monitor, 'entry_events', side_effect=[({}, True), ({key: ('missing_entry', {})}, True)]), \
                 patch.object(monitor, 'notify') as send:
                monitor.watch(root, 'test-job')
                monitor.watch(root, 'test-job')
                send.assert_called_once()
                self.assertIn('Ingresso paper saltato', send.call_args.args[0])

    def test_persistent_failure_alerts_once_after_recovery_grace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'protocol.json').write_text(json.dumps(self.config))
            with patch.object(monitor.time, 'time_ns', return_value=self.anchor * 1000000), \
                 patch.object(monitor.time, 'monotonic', side_effect=[1, 30, 61, 61, 66]), \
                 patch.object(monitor.time, 'sleep'), \
                 patch.object(monitor, 'entry_events', side_effect=[({}, False)] * 4 + [({}, True)]), \
                 patch.object(monitor, 'writer_status', return_value=(None, 'processo assente')), \
                 patch.object(monitor, 'notify') as send:
                monitor.watch(root, 'test-job')
                send.assert_called_once_with('Recupero Timestral fallito. processo assente.')
                state = json.loads((root / 'mac-notification-state.json').read_text())
                self.assertTrue(state['recovery_alerted'])

    def test_first_outcome_waits_for_boundary_quotes(self):
        boundary = self.anchor + 8 * monitor.DAY
        entries = {f'entry:{self.anchor}': ('entry', {}), f'outcome:{self.anchor}': ('outcome', {})}
        with patch.object(monitor, 'audit') as audit:
            self.assertIsNone(monitor.first_week_notice(Path('.'), self.config, entries, boundary + 5000, None))
            status, message = monitor.first_week_notice(Path('.'), self.config, entries, boundary + 120000, None)
            self.assertEqual(status, 'awaiting_quotes')
            self.assertIn('Attendo le quotazioni', message)
            self.assertIsNone(monitor.first_week_notice(Path('.'), self.config, entries, boundary + 125000, status))
            audit.assert_not_called()

    def test_first_week_numbers_retry_failed_delivery_then_deduplicate(self):
        boundary = self.anchor + 8 * monitor.DAY
        entries = {f'entry:{self.anchor}': ('entry', {}), f'outcome:{self.anchor}': ('outcome', {}),
                   f'entry:{self.anchor + monitor.WEEK}': ('entry', {})}
        week = dict(status='valued', quote_ms=boundary+1000, timestral_minus_ewma_pp=-0.20,
                    daily_marks=6, expected_daily_marks=7,
                    arms={'timestral': dict(net_pnl_usdt=-12.50, net_return_pct=-0.125,
                                            observed_max_drawdown_pct=25, risk_halted=True),
                          'ewma': dict(net_pnl_usdt=7.50, net_return_pct=0.075)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'protocol.json').write_text(json.dumps(self.config))
            with patch.object(monitor.time, 'time_ns', return_value=(boundary+2000)*1000000), \
                 patch.object(monitor, 'entry_events', return_value=(entries, True)), \
                 patch.object(monitor, 'audit', return_value={'independent_paper_accounting': {'first_week': week}}), \
                 patch.object(monitor, 'notify', side_effect=[OSError('temporarily unavailable'), None]) as send:
                monitor.watch(root, 'test-job')
                self.assertFalse((root / 'mac-notification-state.json').exists())
                monitor.watch(root, 'test-job')
                monitor.watch(root, 'test-job')
                self.assertEqual(send.call_count, 2)
                message = send.call_args.args[0]
                for phrase in ('17/11 01:00', 'Timestral -12,50 USDT', 'EWMA +7,50 USDT', '-0,20 pp',
                               '25,00%, a o sopra il 25%', '6/7 rilevazioni', 'Arresto per rischio'):
                    self.assertIn(phrase, message)


if __name__ == '__main__':
    unittest.main()
