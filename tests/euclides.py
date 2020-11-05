import unittest

import numpy as np

from structures import IZ


class IntTest(unittest.TestCase):
    low = -5000
    high = 5000
    random_tests = 10_000

    tests = [
        (60, 36, 12, "easy test"),
        (0, 5, 5, "zero works"),
        (7, 0, 7, "zero works"),
        (0, 0, 0, "zero works"),
        (1, 3274, 1, "one works"),
        (5460, 1, 1, "one works"),
        (-60, 36, 12, "negative works"),
        (60, -36, 12, "negative works"),
        (-60, -36, 12, "negative works")
    ]

    def test_gcd(self):

        for a, b, gcd, msg in self.tests:
            self.assertEqual(IZ.gcd(a, b), gcd, msg)

        for _ in range(self.random_tests):
            a, b = np.random.randint(self.low, self.high, (2,), dtype=np.int32)
            self.assertEqual(IZ.gcd(a, b), np.gcd(a, b), "random works")

    def test_bezout(self):

        for a, b, gcd, msg in self.tests:
            g, (x, y) = IZ.bezout(a, b)
            self.assertEqual(g, gcd, "bezout gcd - " + msg)
            self.assertEqual(a * x + b * y, g, f"bezout {a}*{x} + {b}*{y} == {g} - " + msg)

        for _ in range(self.random_tests):
            a, b = np.random.randint(self.low, self.high, (2,), dtype=np.int32)
            g, (x, y) = IZ.bezout(a, b)
            self.assertEqual(g, np.gcd(a, b), "bezout gcd - random works")
            self.assertEqual(a * x + b * y, g, f"bezout {a}*{x} + {b}*{y} == {g} - random works")


if __name__ == '__main__':
    unittest.main()
