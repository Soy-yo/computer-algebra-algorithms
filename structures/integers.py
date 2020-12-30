import numpy as np

from structures.domains import EuclideanDomain
from structures.rings import UnitaryRing, CommutativeRing


class IntegerRing(EuclideanDomain):
    """
    Implementation of the ring of integers IZ := {... -2, -1, 0, 1, 2, ...}.
    """

    def __init__(self):
        super(IntegerRing, self).__init__(int)
        self._factors = None
        self._factor_limit = None

    @property
    def size(self):
        return -1

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

    def divmod(self, a, b):
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

    def eq(self, a, b):
        return a @ self == b @ self

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
        return 1 if a >= 0 else -1

    def is_prime(self, p):
        # TODO use a better algorithm
        return len(self.factor(p)) == 1

    @property
    def factor_limit(self):
        return self._factor_limit

    @factor_limit.setter
    def factor_limit(self, n):
        """
        Sets the factor limit to be able to apply the factor method.
        :param n: int - maximum factorable number
        """
        # Initialization
        n = n @ self
        m = int(np.sqrt(n))
        self._factors = np.zeros(n, dtype=np.int64)
        self._factor_limit = n
        for i in range(1, n, 2):
            self._factors[i] = i
        for i in range(2, n, 2):
            self._factors[i] = 2

        for i in range(3, m):
            if self._factors[i] == i:
                for j in range(i * i, n, i):
                    if self._factors[j] == j:
                        self._factors[j] = i

    def factor(self, a):
        if self.factor_limit is None or self.factor_limit <= a:
            self.factor_limit = 2 * a
        # Computation
        result = []
        while a != 1:
            result.append(self._factors[a])
            a //= self._factors[a]

        return result

    def contains(self, a):
        return isinstance(a, (int, np.integer)) or isinstance(a, (float, np.float)) and a.is_integer()

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
        self._modulo = n

    @property
    def size(self):
        return self._modulo

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
        gcd, (x, _) = IZ.bezout(a_, self._modulo)
        if gcd != 1:
            raise ValueError(f"only units have inverse, but {a} is not an unit")
        return x @ self

    def pow(self, a, n):
        a = a @ self
        return (a ** n) % self._modulo

    def eq(self, a, b):
        return a @ self == b @ self

    def is_zero_divisor(self, a):
        return not self.is_unit(a)

    def contains(self, a):
        return isinstance(a, (int, np.integer)) or isinstance(a, (float, np.float)) and a.is_integer()

    def at(self, a):
        if a in self:
            return int(a) % self._modulo
        raise ValueError(f"{a} is not an element of {self}")

    def __latex__(self):
        return r"\mathbb{Z}_{" + str(self._modulo) + "}"

    def __repr__(self):
        return f"Ring of integers modulo {self._modulo}"
