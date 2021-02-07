import random

import numpy as np

from compalg.domains import EuclideanDomain, PolynomialUFD
from compalg.rings import UnitaryRing, CommutativeRing


class IntegerRing(EuclideanDomain):
    """
    Implementation of the ring of integers IZ := {... -2, -1, 0, 1, 2, ...}.
    """

    def __init__(self):
        super(IntegerRing, self).__init__(int)
        self._factors = None
        self._factor_limit = None

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

    def is_prime(self, p, method='mr', **kwargs):
        """
        Determines whether the element p is prime or not. This can be done through three algorithms: AKS, Miller-Rabin
        or brute force.
        AKS is efficient for detecting composite numbers, but is very slow for detecting big primes.
        Miller-Rabin is a probabilistic algorithm, so it may fail, but this makes it faster in case of success. If p is
        composite, running k iterations will declare p prime with a probability at most 4^-k.
        Brute force uses the prime factor decomposition of the number, which is fast with small numbers, but cannot
        be computed with very large ones as it uses a lot of memory.
        :param p: int - element to be checked
        :param method: str - algorithm to use; one of 'aks' (AKS), 'mr' (Miller-Rabin, default) or 'bf' (Brute Force)
        :param k: int - if method='mr', number of iterations that will be run (default k=100)
        :return: bool - True if p is prime, False otherwise
        """
        from .polynomials import Var, Polynomial

        if method == 'aks':
            # AKS
            for b in range(2, int(np.floor(np.log2(p))) + 1):
                if np.floor(p ** (1 / b)) ** b == p:
                    return False
            stop = False
            r = 1
            while not stop:
                r += 1
                Zr = IZ(r)
                for k in range(1, int(np.floor(np.log2(p) ** 2)) + 1):
                    if (p ** k) @ Zr != Zr.one:
                        stop = True
                        break
            for a in range(2, r + 1):
                if 1 < self.gcd(p, a) < p:
                    return False
            if p <= r:
                return True
            Zp = IZ(p)
            x = Var('x')
            poly_ring = PolynomialUFD(IZ, x)
            for a in range(1, int(np.floor(np.sqrt(IZ(r).totient) * np.log2(p))) + 1):
                f = poly_ring.pow((x + a), p)
                g = x ** r - 1
                h = poly_ring.divmod(f, g)[1]
                h = Polynomial([c @ Zp for c in h.coefficients], h.var)
                if h == x ** p + a:
                    return False

            return True

        if method == 'mr':
            # Source: https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
            # Input must be p > 3
            if p < 0:
                p = -p
            if p == 1:
                return False
            elif p == 2 or p == 3:
                return True
            # Write n as 2^r · d + 1 with d odd
            n = p - 1
            r = 0
            while n % 2 == 0:
                r += 1
                n = n // 2
            d = n
            k = 100 if 'k' not in kwargs else kwargs['k']
            for _ in range(k):
                a = random.randint(2, p - 2)
                x = (a ** d) % p
                if x == 1 or x == p - 1:
                    continue
                finish = False
                for _ in range(0, r - 1):
                    x = (x ** 2) % p
                    if x == p - 1:
                        finish = True
                        break
                if not finish:
                    return False
            return True

        if method == 'bf':
            # Brute force
            return len(self.factor(p)) == 1

        raise ValueError(f"unknown method {method} (expecting one of 'aks', 'mr' or 'bf')")

    def get_random_prime(self, b, **kwargs):
        """
        Miller-Rabin test can be easily used for generating random primes of certain size.
        :param b: int - number of bits of the desired prime
        :param k: int - number of iterations used for Miller-Rabin algorithm
        :return: int - prime number with probability 4^-k (where k=100 is the default value if unspecified)
        """
        while True:
            odd = False
            a = None
            while not odd:
                a = random.randint(2 ** (b - 1), 2 ** b - 1)
                if a % 2 != 0:
                    odd = True
            if self.is_prime(a, method='mr', **kwargs):
                return a

    @property
    def factor_limit(self):
        """
        Factor limit to be able to apply the factor method.
        :return: int - factor limit to be able to apply the factor method
        """
        return self._factor_limit

    @factor_limit.setter
    def factor_limit(self, n):
        """
        Sets the factor limit to be able to apply the factor method.
        :param n: int - maximum factorable number
        """
        # Initialization
        n = n @ self
        self._factors = np.zeros(n, dtype=np.int64)
        self._factor_limit = n
        for i in range(1, n, 2):
            self._factors[i] = i
        for i in range(2, n, 2):
            self._factors[i] = 2

        for i in range(3, n):
            if self._factors[i] == i:
                for j in range(i * i, n, i):
                    if self._factors[j] == j:
                        self._factors[j] = i

    def divisors(self, a, positive=False):
        a = abs(a)
        if positive:
            return [x for x in range(1, a + 1) if self.divides(x, a)]
        ds = self.divisors(a, positive=True)
        return [-d for d in ds] + ds

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

    def __call__(self, n):
        """
        Returns an instance of the ring of integers modulo n (IZ/nIZ).
        :param n: int - modulo of the ring
        :return: ModuloIntegers - ring of integers modulo n instance
        """
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
        """
        :param n: int - modulo of the elements (Z_n = {0, ..., n-1})
        """
        super(ModuloIntegers, self).__init__(int)
        if n <= 1:
            raise ValueError("n must be greater than 1 (IZ/IZ = {0}, so it is not an unitary ring)")
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

    @property
    def totient(self):
        """
        Euler's Phi function Phi(n), where n is the modulo of this ring. This is equivalent to the order of the
        multiplicative group IZ_n or the number of units in this ring.
        :return: int - Phi(n)
        """
        return 1 + sum(1 for i in range(2, self._modulo) if self.is_unit(i))

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
        return pow(a, n, self._modulo)

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

    def __eq__(self, other):
        return isinstance(other, ModuloIntegers) and other._modulo == self._modulo

    def __latex__(self):
        return r"\mathbb{Z}_{" + str(self._modulo) + "}"

    def __repr__(self):
        return f"Ring of integers modulo {self._modulo}"
