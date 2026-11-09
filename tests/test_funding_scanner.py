import unittest

from research.funding_scanner import historical_summary


class FundingHistoryTests(unittest.TestCase):
    def test_hourly_and_eight_hourly_have_same_time_weighted_rate(self):
        hour = 3600000
        hourly = [(i * hour, .00001) for i in range(25)]
        eight_hourly = [(i * 8 * hour, .00008) for i in range(4)]
        self.assertAlmostEqual(historical_summary(hourly)['hist_ann'],
                               historical_summary(eight_hourly)['hist_ann'])

    def test_first_rate_is_outside_observed_accrual_span(self):
        result = historical_summary([(0, 99), (86400000, -.001)])
        self.assertAlmostEqual(result['hist_ann'], -36.5)
        self.assertEqual(result['pct_positive'], 0)

    def test_no_history_and_duplicate_fail_closed(self):
        self.assertEqual(historical_summary([(0, .1)])['n_hist'], 0)
        with self.assertRaises(ValueError):
            historical_summary([(0, .1), (0, .1)])
        with self.assertRaises(ValueError):
            historical_summary([(0, .1), (1, float('nan'))])
