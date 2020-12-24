import unittest

import numpy as np

from structures import IZ


class IntTest(unittest.TestCase):
    low = -5000
    high = 5000
    random_tests = 10_000

    tests = [
        (60, 36, 12, "easy test"),
        (0, 5, 5, "zero test"),
        (7, 0, 7, "zero test"),
        (0, 0, 0, "zero test"),
        (1, 3274, 1, "one test"),
        (5460, 1, 1, "one test"),
        (-60, 36, 12, "negative test"),
        (60, -36, 12, "negative test"),
        (-60, -36, 12, "negative test")
    ]

    def test_gcd(self):

        for a, b, gcd, msg in self.tests:
            self.assertEqual(gcd, IZ.gcd(a, b), msg)

        self.assertEqual(6, IZ.gcd(60, 36, 18), "three test")

        for _ in range(self.random_tests):
            a, b = np.random.randint(self.low, self.high, (2,), dtype=np.int32)
            self.assertEqual(np.gcd(a, b), IZ.gcd(a, b), "random test")

    def test_bezout(self):

        for a, b, gcd, msg in self.tests:
            g, (x, y) = IZ.bezout(a, b)
            self.assertEqual(gcd, g, "bezout gcd - " + msg)
            self.assertEqual(g, a * x + b * y, f"bezout {a}*{x} + {b}*{y} == {g} - " + msg)

        for _ in range(self.random_tests):
            a, b = np.random.randint(self.low, self.high, (2,), dtype=np.int32)
            g, (x, y) = IZ.bezout(a, b)
            self.assertEqual(np.gcd(a, b), g, "bezout gcd - random test")
            self.assertEqual(g, a * x + b * y, f"bezout {a}*{x} + {b}*{y} == {g} - random test")


if __name__ == '__main__':
    unittest.main()
