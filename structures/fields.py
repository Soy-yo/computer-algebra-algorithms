import abc

import numpy as np

from . import domains, polynomials, rings, integers


class Field(rings.DivisionRing, rings.CommutativeRing, domains.Domain, abc.ABC):
    """
    Class representing a field, a commutative ring for which all elements are units but 0.
    """

    def __getitem__(self, var):
        return polynomials.PolynomialField(self, var)


class FiniteField(Field):

    def __init__(self, p, n=1):
        super(FiniteField, self).__init__(polynomials.Polynomial)
        if n <= 0:
            raise ValueError("n must be positive")
        if not integers.IZ.is_prime(p):
            raise ValueError("p must be prime")
        self._base_ring = integers.ModuloIntegers(p)
        self._p = p
        self._n = n

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
        return 0

    @property
    def one(self):
        return 1

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

    # TODO algoretmo
    def inverse(self, a):
        pass

    def eq(self, a, b):
        a = a @ self
        b = b @ self
        return a == b

    def contains(self, a):
        return isinstance(a, (polynomials.Polynomial, int, np.integer))

    # TODO Esta es la mayor gracia
    def at(self, a):
        pass

    def __latex__(self):
        return fr"\mathbb{{F}}_{self.q}"


IF = FiniteField
