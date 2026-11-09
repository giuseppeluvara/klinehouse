import unittest

from research.spot_portfolio import rebalance


class PortfolioTests(unittest.TestCase):
    def test_full_investment_does_not_borrow_fees(self):
        cash, units, cost = rebalance(10000, [0], [100], [1], .001, .0002)
        self.assertEqual(cash, 0)
        self.assertAlmostEqual(units[0], 10000 / (100 * 1.0002 * 1.001))
        self.assertAlmostEqual(units[0]*100 + cost, 10000)

    def test_unwind_and_no_turnover_no_cost(self):
        cash, units, _ = rebalance(10000, [0], [100], [.5], .001, .0002)
        nav = cash + units[0] * 100
        weight = units[0] * 100 / nav
        cash2, units2, cost = rebalance(cash, units, [100], [weight], .001, .0002)
        self.assertAlmostEqual(cost, 0)
        out, empty, _ = rebalance(cash2, units2, [110], [0], .001, .0002)
        self.assertAlmostEqual(out, cash2 + units2[0] * 110 * .9998 * .999)
        self.assertEqual(empty, [0])

    def test_reject_leverage_and_nonfinite(self):
        for weights in ([1.01], [-1], [float('nan')]):
            with self.assertRaises(ValueError):
                rebalance(100, [0], [1], weights, .001, .0002)
