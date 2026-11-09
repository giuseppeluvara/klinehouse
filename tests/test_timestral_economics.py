import math
import unittest

from research.timestral_economics import Trade, exposure, net_return, wealth


class EconomicsTests(unittest.TestCase):
    def test_fee_cash_budget_and_partial_exposure(self):
        trade = Trade(0, 1, 100, 110, .5)
        expected = .5 * (110 * .9998 * .999 / (100 * 1.0002 * 1.001) - 1)
        self.assertAlmostEqual(net_return(trade, .001, .0002), expected)
        self.assertEqual(net_return(Trade(0, 1, 100, 1, 0), .001, .0002), 0)

    def test_cost_hurdle_and_unknown_policy(self):
        levels = [.05, .25, .5, .95]
        q = [-.01, .001, .002, .02]
        self.assertEqual(exposure('median_positive', levels, q, 4, .001, .0002), 1)
        self.assertEqual(exposure('lower_quartile_above_cost', levels, q, 4, .001, .0002), 0)
        with self.assertRaises(ValueError):
            exposure('future_tuned_rule', levels, q, 4, .001, .0002)
        with self.assertRaises(ValueError):
            exposure('median_positive', levels, [0, -.1, 0, .1], 4, .001, .0002)

    def test_compounding_and_drawdown(self):
        total, dd = wealth([.1, -.2, .25])
        self.assertAlmostEqual(total, .1)
        self.assertAlmostEqual(dd, .2)
        with self.assertRaises(ValueError):
            wealth([math.nan])


if __name__ == '__main__':
    unittest.main()
