import warnings

import numpy as np

from .domains import EuclideanDomain
from .rings import UnitaryRing, CommutativeRing


class IntegerRing(EuclideanDomain):
    """
    Implementation of the ring of integers IZ := {... -2, -1, 0, 1, 2, ...}.
    """

    def __init__(self):
        super(IntegerRing, self).__init__(int)

    @property
    def char(self):
        return 0

    @property
    def zero(self):
        return 0

    @property
    def one(self):
        return 1

    def add(self, a, b):
        a = a @ self
        b = b @ self
        return a + b

    def sub(self, a, b):
        a = a @ self
        b = b @ self
        return a - b

    def mul(self, a, b):
        a = a @ self
        b = b @ self
        return a * b

    def full_div(self, a, b):
        # TODO check // and % do what we want (specially for negative integers)
        a = a @ self
        b = b @ self
        return a // b, a % b

    def negate(self, a):
        a = a @ self
        return -a

    def times(self, a, n):
        # We allow n < 0 here meaning self.times(self.negate(a), n)
        return self.mul(a, n)

    def is_unit(self, a):
        a = a @ self
        return a in (1, -1)

    def inverse(self, a):
        a = a @ self
        if not self.is_unit(a):
            raise ValueError(f"only units have inverse, but {a} is not an unit")
        # Only units are {-1, 1} and their inverses are -1 and 1, respectively
        return a

    def pow(self, a, n):
        a = a @ self
        if not isinstance(n, int):
            raise ValueError(f"expected n to be an integer but found {n}: {type(n)}")
        if n < 0:
            a = self.inverse(a)
            n = -n
        return a ** n

    def is_idempotent(self, a):
        a = a @ self
        # Only idempotent elements are 0 and 1
        return a in (0, 1)

    def value(self, a):
        a = a @ self
        return abs(a)

    def normal_part(self, a):
        a = a @ self
        return abs(a)

    def unit_part(self, a):
        a = a @ self
        return 1 if a > 0 else -1 if a < 0 else 0

    def is_prime(self, p):
        pass

    def factor(self, a):
        pass

    def contains(self, a):
        return isinstance(a, (int, np.integer))

    def at(self, a):
        if a in self:
            return int(a)
        raise ValueError(f"{a} is not an element of {self}")

    def __latex__(self):
        return r"\mathbb{Z}"

    # TODO return F_n if n is prime and some trivial ring if n == 0
    def __call__(self, n):
        """
        Returns an instance of the ring of integers modulo n (IZ/nIZ).
        :param n: int - modulo of the ring
        :return: ModuloIntegers or FiniteField - ring of integers modulo n instance
        """
        if n == 0:
            pass
        if self.is_prime(n):
            pass
        return ModuloIntegers(n)

    def __repr__(self):
        return "Ring of integers"


# We will only export one instance of the ring of integers
IZ = IntegerRing()


class ModuloIntegers(UnitaryRing, CommutativeRing):
    """
    Implementation of the ring of integers modulo n IZ_n = IZ/nIZ := {[0], [1], ..., [n-1]}, where
    [a] := {ka : k in IZ}.
    The main operations are defined as follows:
        - [a] + [b] = [a + b].
        - [a] * [b] = [a * b].
    """

    def __init__(self, n):
        super(ModuloIntegers, self).__init__(int)
        if n <= 1:
            raise ValueError("n must be a greater than 1 (IZ/IZ = {0}, so it is not an unitary ring)")
        if IZ.is_prime(n):
            warnings.warn(
                "It is not recommended to use ModuloIntegers when n is prime "
                "as it is actually a field and implements more methods"
            )
        self._modulo = n

    @property
    def char(self):
        return self._modulo

    @property
    def zero(self):
        return 0

    @property
    def one(self):
        return 1

    def add(self, a, b):
        a = a @ self
        b = b @ self
        return (a + b) % self._modulo

    def sub(self, a, b):
        a = a @ self
        b = b @ self
        return (a - b) % self._modulo

    def mul(self, a, b):
        a = a @ self
        b = b @ self
        return (a * b) % self._modulo

    def negate(self, a):
        a = a @ self
        return self._modulo - a

    def times(self, a, n):
        # We allow n < 0 here meaning self.times(self.negate(a), n)
        return self.mul(a, n)

    def is_unit(self, a):
        a = a @ self
        return a != 0 and IZ.are_coprime(a, self._modulo)

    def inverse(self, a):
        # ax + ny = 1 => ax - 1 = (-y)n => ax ≡ 1 mod n
        a_ = a @ self
        (x, _), gcd = IZ.bezout(a_, self._modulo)
        if gcd != 1:
            raise ValueError(f"only units have inverse, but {a} is not an unit")
        return x

    def pow(self, a, n):
        a = a @ self
        return (a ** n) % self._modulo

    def is_zero_divisor(self, a):
        return not self.is_unit(a)

    def contains(self, a):
        return isinstance(a, (int, np.integer))

    def at(self, a):
        if a in self:
            return int(a) % self._modulo
        raise ValueError(f"{a} is not an element of {self}")

    def __latex__(self):
        return r"\mathbb{Z}_{" + str(self._modulo) + "}"

    def __repr__(self):
        return f"Ring of integers modulo {self._modulo}"
