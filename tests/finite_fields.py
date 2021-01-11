import random
import unittest

from structures import IF, Var


class FiniteFieldsTest(unittest.TestCase):

    def test_inverse(self):
        field = IF(3)

        self.assertEqual(1, field.inverse(1), "easy test")
        self.assertEqual(2, field.inverse(2), "easy test")
        self.assertRaises((ZeroDivisionError, ValueError), field.inverse, 0)

        mods = [17, 19, 23, 37, 97, 773]
        for p in mods:
            field = IF(p)
            for _ in range(20):
                a = random.randrange(1, p)
                self.assertEqual(pow(a, -1, p), field.inverse(a), "random test")

        x = Var('x')
        field = IF(2, x ** 8 + x ** 4 + x ** 3 + x + 1)
        a = x ** 6 + x ** 4 + x + 1
        self.assertEqual(x ** 7 + x ** 6 + x ** 3 + x, field.inverse(a), "poly test")
        self.assertRaises((ZeroDivisionError, ValueError), field.inverse, 0)

    def test_irreducibility(self):
        t = Var('t')
        x = Var('x')
        field = IF(2)[t]

        self.assertTrue(field.is_irreducible(t + 1), "easy test True")
        self.assertTrue(field.is_irreducible(t ** 7 + t ** 6 + t ** 5 + t ** 4 + t ** 2 + t + 1), "easy test True")

        field = IF(3)[t]
        self.assertFalse(field.is_irreducible(t ** 3 + 1), "easy test False")  # (t + 1) ** 3

        field = IF(2, x ** 2 + x + 1)[t]
        self.assertTrue(field.is_irreducible(t ** 2 - t + x), "medium test True")
        self.assertFalse(field.is_irreducible(t ** 2 - (x + 1)), "medium test False")  # (t - x) * (t + x)

        field = IF(3, x ** 2 - x - 1)[t]
        # TODO too slow :(
        # self.assertFalse(field.is_irreducible(t ** 3 + t ** 2 + t - x), "medium test False")
        # (t + x) * (t ** 2 - (x - 1) * t - 1)
        self.assertTrue(field.is_irreducible(t ** 2 - t * (x - 1) - 1), "medium test True")

    def test_logarithm(self):
        # g^x = h => x = log(g, h)
        field = IF(5)
        self.assertEqual(3, field.discrete_logarithm(2, 3), "easy test")

        field = IF(59)
        self.assertEqual(25, field.discrete_logarithm(2, 11), "medium test")

        field = IF(383)
        self.assertEqual(110, field.discrete_logarithm(2, 228), "medium test")

        x = Var('x')
        field = IF(2, x ** 2 + x + 1)
        self.assertEqual(2, field.discrete_logarithm(x, x + 1), "hard test")

        field = IF(3, x ** 2 + 1)
        self.assertEqual(7, field.discrete_logarithm(x + 1, x + 2), "hard test")

        # TODO apparently we have to do it in IF_q[x]/(f(x)) but I think that it makes no sense

    def test_factor(self):
        t = Var('t')
        x = Var('x')
        field = IF(2)[t]

        self.assertDictEqual({2: t + 1 @ field}, field.square_free_factor(t ** 2 + 1), "easy test")

        f = (t + 1) ** 2 * t ** 3
        self.assertDictEqual({3: t @ field, 2: t + 1 @ field}, field.square_free_factor(f), "easy test")

        field = IF(5)[t]
        g = t
        h = t + 3
        f = g ** 3 * h ** 3
        self.assertDictEqual({3: g * h}, field.square_free_factor(f), "medium test")

        field = IF(7, x ** 3 + x + 1)[t]
        f1 = t
        f2 = t ** 2 + 2
        f3 = t ** 3 + t * (x + 1) + (x ** 2)
        f = f1 ** 3 * f2 ** 2 * f3 ** 3
        self.assertDictEqual({2: f2, 3: f1 * f3}, field.square_free_factor(f), "hard test")


if __name__ == '__main__':
    unittest.main()
