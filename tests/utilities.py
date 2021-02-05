import unittest

from algorithms import *
from structures import link, set_ordering, IF


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

    def test_groebner(self):
        x, y = link('x', 'y')
        field = IF(7)

        set_ordering('dp')

        f1 = (x ** 3) @ field
        f2 = (x ** 2 * y + 6 * y ** 4) @ field
        self.assertCountEqual([f1, f2], groebner_basis([f1, f2], field), "easy test")

        set_ordering('lp')

        f1 = (x ** 3) @ field
        f2 = (x ** 2 * y + 6 * y ** 4) @ field
        f3 = (x * y ** 4) @ field
        f4 = (y ** 7) @ field
        self.assertCountEqual([f1, f2, f3, f4], groebner_basis([f1, f2], field), "medium test")

        x, y, z = link('x', 'y', 'z')
        f1 = (x * y - 1) @ field
        f2 = (x * z - 1) @ field
        f3 = (y - z) @ field
        self.assertCountEqual([f1, f2, f3], groebner_basis([f1, f2], field), "medium test")

        f1 = (z ** 2 - 2) @ field
        f2 = (y ** 2 + 2 * y - 1) @ field
        f3 = ((y + z + 1) * x + y * z + z + 2) @ field
        f4 = (x ** 2 + x + y - 1) @ field
        self.assertCountEqual([f1, f2, f3, f4], groebner_basis([f1, f2, f3, f4], field), "medium test")

    def test_ideal(self):
        x, y = link('x', 'y')
        field = IF(7)

        set_ordering('dp')

        f1 = (x ** 3) @ field
        f2 = (x ** 2 * y + 6 * y ** 4) @ field
        self.assertTrue(in_ideal(f1, [f1, f2], field), "easy test")
        self.assertTrue(in_ideal(f2, [f1, f2], field), "easy test")
        self.assertTrue(in_ideal(3 * y ** 2 * f1 + x * f2, [f1, f2], field), "easy test")
        self.assertFalse(in_ideal(f1 + f2 + x, [f1, f2], field), "medium test")

        set_ordering('lp')

        self.assertTrue(in_ideal((y ** 7) @ field, [f1, f2], field), "medium test")
        self.assertTrue(in_ideal(f1 + f2 + y ** 7, [f1, f2], field), "medium test")

        x, y, z = link('x', 'y', 'z')
        f1 = (x * y - 1) @ field
        f2 = (x * z - 1) @ field
        self.assertFalse(in_ideal(2 * f1 + z * f2 + (x + y) * (y - z) + y ** 2, [f1, f2], field), "medium test")


if __name__ == '__main__':
    unittest.main()
