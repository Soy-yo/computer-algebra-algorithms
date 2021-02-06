import unittest

from compalg import IZ, Var


class IntegerFactorization(unittest.TestCase):

    def test_poly_factorization(self):
        x = Var('x')
        ring = IZ[x]
        methods = ['km', 'hl']

        for method in methods:

            f = x + 1
            self.assertCountEqual([f], ring.factor(f, method=method), "easy test")

            if method != 'hl':
                f = x ** 2 + 2 * x + 1
                self.assertCountEqual([x + 1, x + 1], ring.factor(f, method=method), "easy test")

                f = 2 * x ** 4 + 4 * x ** 2 + 2
                g = x ** 2 + 1
                self.assertCountEqual([g, g, 2], ring.factor(f, method=method), "medium test")

            g = x ** 2 + 1
            h = x ** 4 + x ** 2 + 10
            f = g * h
            self.assertCountEqual([g, h], ring.factor(f, method=method), "medium test")

            f = x ** 7 + 2 * x ** 5 + x ** 4 - 2 * x ** 3 + x ** 2 - 3 * x - 3
            g = x ** 3 + x + 1
            h = x ** 4 + x ** 2 - 3
            self.assertCountEqual([g, h], ring.factor(f, method=method), "hard test")


if __name__ == '__main__':
    unittest.main()
