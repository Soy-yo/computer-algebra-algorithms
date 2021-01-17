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

    def test_sf_factor(self):
        t = Var('t')
        x = Var('x')
        field = IF(2)[t]

        self.assertCountEqual([(t + 1 @ field, 2)], field.square_free_factorization(t ** 2 + 1), "easy test")

        f = (t + 1) ** 2 * t ** 3
        self.assertCountEqual([(t @ field, 3), (t + 1 @ field, 2)], field.square_free_factorization(f), "easy test")

        field = IF(5)[t]
        g = t @ field
        h = (t + 3) @ field
        f = (g ** 3 * h ** 3) @ field
        self.assertCountEqual([(g * h @ field, 3)], field.square_free_factorization(f), "medium test")

        field = IF(7, x ** 3 + x + 1)[t]
        f1 = t @ field
        f2 = (t ** 2 + 2) @ field
        f3 = (t ** 3 + t * (x + 1) + (x ** 2)) @ field
        f = (f1 ** 3 * f2 ** 2 * f3 ** 3) @ field
        self.assertCountEqual([(f2, 2), (f1 * f3, 3)], field.square_free_factorization(f), "hard test")

    def test_dd_factor(self):
        t = Var('t')
        x = Var('x')
        field = IF(2)[t]

        self.assertCountEqual([(t + 1 @ field, 1)], field.distinct_degree_factorization(t + 1), "easy test")

        f = (t ** 3 + t + 1) @ field
        self.assertCountEqual([(f, 3)], field.distinct_degree_factorization(f), "easy test")

        g = (t + 1) @ field
        h = (f * g) @ field
        self.assertCountEqual([(f, 3), (g, 1)], field.distinct_degree_factorization(h), "easy test")

        h1 = (t ** 7 + t ** 6 + t ** 5 + t ** 4 + t ** 2 + t + 1) @ field
        # "Random" polynomial, but it's irreducible
        h2 = (t ** 7 + t ** 5 + t ** 4 + t ** 3 + t ** 2 + t + 1) @ field
        self.assertCountEqual([(f, 3), (g, 1), ((h1 * h2) @ field, 7)],
                              field.distinct_degree_factorization((f * g * h1 * h2) @ field),
                              "medium test")

        field = IF(2, x ** 3 + x + 1)[t]
        f1 = t @ field
        # "Random" polynomials, but they are irreducible
        f2 = (t ** 3 + t ** 2 + (x ** 2 + x)) @ field
        f3 = (t ** 3 + t * (x + 1) + (x ** 2)) @ field
        f = (f1 * f2 * f3) @ field
        self.assertCountEqual([(f1, 1), ((f2 * f3) @ field, 3)], field.distinct_degree_factorization(f), "hard test")

    def test_ed_factor(self):
        t = Var('t')
        x = Var('x')
        field = IF(2)[t]

        self.assertCountEqual([(t + 1) @ field], field.equal_degree_factorization(t + 1, 1), "easy test")

        f = (t ** 2 + 1) @ field
        self.assertCountEqual([(t + 1) @ field, (t + 1) @ field], field.equal_degree_factorization(f, 1), "easy test")

        g = (t ** 3 + t + 1) @ field
        h = (t ** 3 + t ** 2 + 1) @ field
        f = (g * h) @ field
        self.assertCountEqual([g, h], field.equal_degree_factorization(f, 3), "medium test")

        field = IF(7)[t]
        g = (t ** 3 + 5 * t + 5) @ field
        h = (t ** 3 + 4 * t ** 2 + 3) @ field
        f = (g * h) @ field
        self.assertCountEqual([g, h], field.equal_degree_factorization(f, 3), "medium test")

        field = IF(2, x ** 3 + x + 1)[t]
        f1 = (t ** 2 + t + x + 1) @ field
        f2 = (t ** 2 * x + t * (x + 1) + x) @ field
        f = (f1 * f2) @ field
        self.assertCountEqual([f1, f2], field.equal_degree_factorization(f, 2), "hard test")

    def test_factor(self):
        for method in ['ts', 'bfa']:
            t = Var('t')
            x = Var('x')
            field = IF(2)[t]

            f = (t + 1) @ field
            self.assertCountEqual([f], field.factor(f, method=method), "easy test " + method)

            g = (t + 1) @ field
            h = t @ field
            f = (g * h) @ field
            self.assertCountEqual([g, h], field.factor(f, method=method), "easy test " + method)

            f = (g ** 2) @ field
            self.assertCountEqual([g, g], field.factor(f, method=method), "easy test " + method)

            field = IF(5)[t]
            g = (2 * t + 1) @ field
            h = field.divmod(g, 2)[0]
            f = (g ** 2) @ field
            self.assertCountEqual([h, h, 4], field.factor(f, method=method), "medium test " + method)

            g = (t ** 3 + t + 1) @ field
            h = (t + 2) @ field
            f = (g ** 2 * h ** 2) @ field
            self.assertCountEqual([g, g, h, h], field.factor(f, method=method), "medium test " + method)

            field = IF(2, x ** 3 + x + 1)[t]
            f1 = (t + 1) @ field
            f2 = (t + x + 1) @ field
            f3 = (t ** 2 + t * x + x) @ field
            f = (f1 ** 3 * f2 ** 2 * f3) @ field
            self.assertCountEqual([f1, f1, f1, f2, f2, f3], field.factor(f, method=method), "hard test " + method)


if __name__ == '__main__':
    unittest.main()
