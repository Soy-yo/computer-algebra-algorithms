import unittest

from structures import Var, IZ, IF


class GCDTest(unittest.TestCase):

    def test_ufd(self):
        x = Var('x')

        ufd = IZ[x]

        p = x ** 2 + 2 * x + 1
        q = x + 1
        self.assertEqual(q, ufd.gcd(p, q), "easy test")

        gcd = x ** 3 + 1
        p = gcd * (6 * x ** 2 + 5 * x + 1)
        q = gcd * (7 * x ** 5 + x ** 2 + 3 * x)
        self.assertEqual(gcd, ufd.gcd(p, q), "medium test")

        p = 3 * (x + 1) * (x - 2) * (x + 5) * (x - 11)
        q = 5 * (x + 3) * (x - 7) * x
        self.assertEqual(-1, ufd.gcd(p, q), "coprime test")

        p = 10 * (x + 1) * (x - 2) * (x + 5) * (x - 11)
        q = 8 * (x + 3) * (x - 7) * x
        self.assertEqual(-2, ufd.gcd(p, q), "almost coprime test")

        a = x ** 3 + 5 * x + 2
        p = 12 * a
        q = 8 * a
        self.assertEqual(4 * a, ufd.gcd(p, q), "same base test")

    def test_field(self):
        t = Var('t')

        field = IF(7)[t]

        p = x ** 2 + 2 * x + 1
        q = x + 1
        self.assertEqual(q, field.gcd(p, q), "easy test")


if __name__ == '__main__':
    unittest.main()
