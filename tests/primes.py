import unittest

from structures import IZ


class PrimalityTest(unittest.TestCase):

    def test_aks(self):
        method = 'aks'
        self.assertTrue(IZ.is_prime(2, method=method), "easy test")
        self.assertTrue(IZ.is_prime(3, method=method), "easy test")
        self.assertTrue(IZ.is_prime(11, method=method), "easy test")
        self.assertTrue(IZ.is_prime(17, method=method), "easy test")
        self.assertTrue(IZ.is_prime(229, method=method), "medium test")
        self.assertFalse(IZ.is_prime(230, method=method), "medium test")
        self.assertFalse(IZ.is_prime(18354, method=method), "medium test")


if __name__ == '__main__':
    unittest.main()
