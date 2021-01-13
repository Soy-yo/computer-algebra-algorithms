import unittest

from algorithms import *


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


if __name__ == '__main__':
    unittest.main()
