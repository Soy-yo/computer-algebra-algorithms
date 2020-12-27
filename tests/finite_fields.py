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
        # self.assertTrue(field.is_irreducible(t ** 8 + t ** 4 + t ** 3 + t + 1), "easy test True")

        field = IF(7)[t]
        self.assertFalse(field.is_irreducible((t + 1) ** 7), "easy test False")

        field = IF(5, x ** 2 + 1)[t]
        self.assertTrue(field.is_irreducible(t ** 4 + t ** 3 + t ** 2 + t + 1), "medium test True")

        field = IF(2, x ** 2 + 1)[t]
        self.assertTrue(field.is_irreducible(t ** 2 - t + x), "medium test True")
        self.assertFalse(field.is_irreducible(t ** 2 - t + (x + 1)), "medium test False")


if __name__ == '__main__':
    unittest.main()
