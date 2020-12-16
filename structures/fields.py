import abc

import numpy as np

from structures.domains import Domain
from structures.integers import IZ, ModuloIntegers
from structures.rings import DivisionRing, CommutativeRing
from structures.polynomials import Var, Polynomial


class Field(DivisionRing, CommutativeRing, Domain, abc.ABC):
    """
    Class representing a field, a commutative ring for which all elements are units but 0.
    """

    def __getitem__(self, var):
        return PolynomialField(type(self), var)


class FiniteField(Field):

    # TODO comprobar irreduciblidad o generar alguno si hace falta
    def __init__(self, p, n=1, base_poly=None):
        super(FiniteField, self).__init__(Polynomial)
        if n <= 0:
            raise ValueError("n must be positive")
        if not IZ.is_prime(p):
            raise ValueError("p must be prime")
        if base_poly is not None:
            self._base_ring = ModuloIntegers(p)[base_poly.var]
        elif n == 1:
            self._base_ring = ModuloIntegers(p)['']
        else:
            raise ValueError("a modulo polynomial must be specified for IF(p, n>1)")
        self._p = p
        self._n = n
        self._base_poly = base_poly if base_poly is not None else Polynomial([0, 1], '')

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
    def size(self):
        return self.q

    @property
    def char(self):
        return self._p

    @property
    def zero(self):
        return Polynomial([0], self._base_ring.var, dtype=int)

    @property
    def one(self):
        return Polynomial([1], self._base_ring.var, dtype=int)

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

    def discrete_logarithm(self, h, g):
        """TODO"""

        def get_s(chi_, k):
            # Get coefficient a_{n-1} -> get term n - 1 as Polynomial and get its last coefficient
            return lambda x: x.term(self.n - 1).coefficients[-1] in range(chi_ * k, chi_ * (k + 1))

        h = h @ self
        g = g @ self
        n = self.q - 1
        Zn = IZ(n)
        chi = np.ceil(self.p / 3)
        # s3 = otherwise
        s1, s2 = get_s(chi, 0), get_s(chi, 1)
        x, a, b = self.one, Zn.zero, Zn.zero
        xs = [x]
        as_ = [a]
        bs = [b]
        while True:
            # it k -> x_k = g^{a_k} * h^{b_k}
            for _ in range(2):
                x, a, b = (self.mul(h, x), a, Zn.add(b, Zn.one)) if s1(x) else \
                    (Zn.pow(x, 2), Zn.times(a, 2), Zn.times(b, 2)) if s2(x) else \
                        (self.mul(g, x), Zn.add(a, Zn.one), b)
                xs.append(x)
                as_.append(a)
                bs.append(b)

            # x_k == x_{k/2}
            if xs[len(xs) // 2] == x and Zn.is_unit(Zn.sub(b - bs[len(bs) // 2])):
                break

        return Zn.mul(Zn.sub(a - as_[len(as_) // 2]), Zn.inverse(Zn.sub(b - bs[len(bs) // 2]))) % self.p

    def eq(self, a, b):
        a = a @ self
        b = b @ self
        return a == b

    def contains(self, a):
        return isinstance(a, (Polynomial, int, np.integer))

    def at(self, a):
        return self._base_ring.divmod(a, self._base_poly)[1] @ self._base_ring

    def __latex__(self):
        return fr"\mathbb{{F}}_{self.q}"


IF = FiniteField


class PolynomialField(Field):

    def __init__(self, base_ring, var):
        super(PolynomialField, self).__init__(base_ring)
        self._base_ring = base_ring
        self._var = Var(var.x) if isinstance(var, Var) else Var(var)

    @property
    def zero(self):
        return self._base_ring.zero

    @property
    def one(self):
        return self._base_ring.one

    @property
    def char(self):
        return self._base_ring.char

    def add(self, a, b):
        a = a @ self
        b = b @ self
        return a + b

    def mul(self, a, b):
        a = a @ self
        b = b @ self
        return a * b

    def negate(self, a):
        a = a @ self
        return -a

    def eq(self, a, b):
        a = a @ self
        b = b @ self
        return a == b

    def is_unit(self, a):
        pass

    # TODO ???
    def inverse(self, a):
        pass

    def is_zero_divisor(self, a):
        pass

    def contains(self, a):
        return a in self._base_ring or all(ai in self._base_ring for ai in a.coefficients)

    def is_irreducible(self, p):
        def poly(k):
            coeffs = np.zeros(k + 1)
            coeffs[1] = -1
            coeffs[-1] = 1
            return Polynomial(coeffs, self._var)

        p = p @ self
        if isinstance(self._base_ring, IF):
            n = self._base_ring.n
            q = self._base_ring.q
            if self.divmod(poly(q), p)[1] != 0:
                return False

            factors = set(IZ.factor(n))
            for r in factors:
                if self.gcd(p, poly(n / r)) != 1:
                    return False

            return True

        raise ValueError(f"cannot check irreducibility in {self._base_ring}")

    def is_prime(self, p):
        return self.is_irreducible(p)

    def factor(self, a):
        pass

    def normal_part(self, a):
        a = a @ self
        u = self.unit_part(a)
        return self.divmod(a, u)[0]

    def unit_part(self, a):
        a = a @ self
        return self._base_ring.unit_part(a.coefficients[-1])

    # GCD de los coeficientes
    def content(self, a):
        a @ self
        return 1

    # a = u(a) * cont(a) * pp(a)
    def primitive_part(self, a):
        a = a @ self
        return self.normal_part(a)

    # Divides a(x) / b(x) and returns it's quotient and remainder
    def divmod(self, a, b):
        """
        :param a:
        :param b:
        :return:
        """
        a = a @ self
        b = b @ self

        if a.var != b.var:
            raise ValueError("variables must be the same")

        quotient, remainder = self.zero, a

        while remainder.degree() >= b.degree:
            monomial_exponent = remainder.degree() - b.degree
            monomial_zeros = [self.zero for _ in range(monomial_exponent)]
            monomial_divisor = Polynomial(monomial_zeros + [remainder.coefficients[-1] / b.coefficients[-1]], a.var)

            quotient += monomial_divisor
            remainder -= monomial_divisor * b

        return quotient, remainder

    def gcd(self, a, b, *args):
        c = self.primitive_part(a)
        d = self.primitive_part(b)

        while d != self.zero:
            r = self.divmod(c, d)[1]
            c = d
            d = self.primitive_part(r)

        gamma = self._base_ring.gcd(self.content(a), self.content(b))
        g = self.mul(gamma, c)
        if args:
            return self.gcd(g, args[0], *args[1:])
        return g

    def at(self, a):
        if a in self._base_ring:
            return Polynomial([a @ self._base_ring], self._var)
        if a not in self:
            raise ValueError("the element must be a polynomial")
        coeffs = a.coefficients
        return Polynomial([ai @ self._base_ring for ai in coeffs], self._var, coeffs.dtype)

    def __latex__(self):
        return self._base_ring.__latex__() + "[" + self._var.__latex__() + "]"
