import unittest

from algorithms import *
from structures import link, IF


class UtilitiesTest(unittest.TestCase):

    def test_congruences(self):
        x, z = solve_congruences([18], [8])
        self.assertEqual((2, 8), (x, z.char), "easy test")

        x, z = solve_congruences([1, 1, 1], [7, 13, 11])
        self.assertEqual((1, 7 * 13 * 11), (x, z.char), "easy test")

        x, z = solve_congruences([4, 12, 0], [9, 14, 17])
        self.assertEqual((544, 9 * 14 * 17), (x, z.char), "medium test")

        x, z = solve_congruences([5], [3], [21])
        self.assertEqual((9, 21), (x, z.char), "medium test")

        x, z = solve_congruences([4, 9, 33, 1], [8, 23, 1, 42], [21, 38, 53, 55])
        self.assertEqual((1153007, 21 * 38 * 53 * 55), (x, z.char), "hard test")

        # ns are not coprime
        self.assertRaises(ValueError, solve_congruences, [0, 0, 0], [12, 35, 21])

    def test_multidivision(self):
        x, = link('x')
        field = IF(2)
        f = x ** 2 + x + 1
        fs = [x + 1]
        q, r = multidivision(f, fs, field)
        self.assertEqual(f @ field, (q * fs[0] + r) @ field, "easy test")

        x, y = link('x', 'y')
        f = x ** 3 * y ** 2
        fs = [x ** 3, x ** 2 * y - y ** 4]
        q1, q2, r = multidivision(f, fs, field)
        self.assertEqual(f @ field, (q1 * fs[0] + q2 * fs[1] + r) @ field, "medium test")

        x, y, z = link('x', 'y', 'z')
        field = IF(17)
        f = x ** 2 * z + 2 * x ** 2 * y + 5 * z ** 2 * x + 3 * y ** 2 * z
        fs = [x ** 2, x * y ** 2, z - y]
        q1, q2, q3, r = multidivision(f, fs, field)
        self.assertEqual(f @ field, (q1 * fs[0] + q2 * fs[1] + q3 * fs[2] + r) @ field, "hard test")


if __name__ == '__main__':
    unittest.main()
