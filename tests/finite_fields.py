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

    # Quizás sirva en un futuro:
    # https://math.stackexchange.com/questions/3053365/discrete-logarithm-problem-in-finite-fields
    def test_logarithm(self):
        field = IF(5)
        # g^x = h => x = log(h, g)
        self.assertEqual(0, field.discrete_logarithm(1, 2), "easy test")
        self.assertEqual(1, field.discrete_logarithm(2, 2), "easy test")
        self.assertEqual(3, field.discrete_logarithm(3, 2), "easy test")

        mods = [17, 19, 23, 37, 97, 773]
        for p in mods:
            field = IF(p)
            for x in range(2, p):
                for _ in range(10):
                    n = random.randrange(1, p)
                    a = pow(x, n, mod=p)
                    self.assertEqual(n, field.discrete_logarithm(a, x), "medium test")


if __name__ == '__main__':
    unittest.main()
