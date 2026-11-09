import unittest
from decimal import Decimal as D

from research.carry_accounting import Costs, Prices, Settlement, account


class CarryAccountingTests(unittest.TestCase):
    def test_basis_loss_can_overwhelm_positive_funding(self):
        result = account(Prices(0, D(100), D(101)),
                         Prices(86400000, D(200), D(205)), D(2),
                         [Settlement(1, D('.01'), D(150))],
                         Costs(D('.001'), D('.0005'), D(1), D('.05')), D(400))
        self.assertEqual(result['funding'], D(3))
        self.assertEqual(result['basis_pnl'], D(-8))
        self.assertEqual(result['fees'], D('.906'))
        self.assertEqual(result['slippage'], D('.1212'))
        self.assertLess(result['net'], D(-6))

    def test_boundary_and_negative_settlements(self):
        result = account(Prices(0, D(100), D(100)), Prices(10, D(100), D(100)),
                         D(1), [Settlement(t, D('-.01'), D(100)) for t in (0, 5, 10)],
                         Costs(D(0), D(0), D(0), D(0)), D(200))
        self.assertEqual(result['net'], D(-1))
        self.assertEqual(result['settlements'], 1)

    def test_reject_duplicate_and_nonfinite(self):
        with self.assertRaises(ValueError):
            Prices(0, D('NaN'), D(1))
        with self.assertRaises(ValueError):
            Costs(D(-1), D(0), D(0), D(0))
        with self.assertRaises(ValueError):
            account(Prices(0, D(100), D(100)), Prices(10, D(100), D(100)), D(1),
                    [Settlement(5, D(0), D(100))] * 2,
                    Costs(D(0), D(0), D(0), D(0)), D(200))


if __name__ == '__main__':
    unittest.main()
