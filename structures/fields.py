import abc

import numpy as np

import structures


class Field(structures.rings.DivisionRing, structures.rings.CommutativeRing, structures.domains.Domain, abc.ABC):
    """
    Class representing a field, a commutative ring for which all elements are units but 0.
    """

    def __getitem__(self, var):
        return structures.polynomials.PolynomialField(self, var)


class FiniteField(Field):

    # TODO comprobar irreduciblidad o generar alguno si hace falta
    def __init__(self, p, n=1, base_poly=None):
        super(FiniteField, self).__init__(structures.polynomials.Polynomial)
        if n <= 0:
            raise ValueError("n must be positive")
        if not structures.integers.IZ.is_prime(p):
            raise ValueError("p must be prime")
        if base_poly is not None:
            self._base_ring = structures.integers.ModuloIntegers(p)[base_poly.var]
        elif n == 1:
            self._base_ring = structures.integers.ModuloIntegers(p)['']
        else:
            raise ValueError("a modulo polynomial must be specified for IF(p, n>1)")
        self._p = p
        self._n = n
        self._base_poly = base_poly if base_poly is not None else structures.polynomials.Polynomial([0, 1], '')

    @property
    def p(self):
        return self._p

    @property
    def n(self):
        return self._n

    @property
    def q(self):
        return self._p ** self._n

    @property
    def zero(self):
        return structures.polynomials.Polynomial([0], self._base_ring.var, dtype=int)

    @property
    def one(self):
        return structures.polynomials.Polynomial([1], self._base_ring.var, dtype=int)

    @property
    def char(self):
        return self._p

    def add(self, a, b):
        a = a @ self
        b = b @ self
        return (a + b) @ self

    def negate(self, a):
        a = a @ self
        return (-a) @ self

    def mul(self, a, b):
        a = a @ self
        b = b @ self
        return (a * b) @ self

    def inverse(self, a):
        # a(t)x + q(t)y = 1 => a(t)x - 1 = (-y)q(t) => a(t)x ≡ 1 mod q(t)
        a_ = a @ self
        gcd, (x, _) = self._base_ring.bezout(a_, self._base_poly)
        if gcd != 1:
            raise ValueError(f"only units have inverse, but {a} is not an unit")
        return x @ self

    def eq(self, a, b):
        a = a @ self
        b = b @ self
        return a == b

    def contains(self, a):
        return isinstance(a, (structures.polynomials.Polynomial, int, np.integer))

    def at(self, a):
        return self._base_ring.divmod(a, self._base_poly)[1] @ self._base_ring

    def __latex__(self):
        return fr"\mathbb{{F}}_{self.q}"


IF = FiniteField
