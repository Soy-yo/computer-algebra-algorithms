import abc

import numpy as np

from structures.domains import Domain
from structures.integers import IZ, ModuloIntegers
from structures.polynomials import Var, Polynomial
from structures.rings import DivisionRing, CommutativeRing


class Field(DivisionRing, CommutativeRing, Domain, abc.ABC):
    """
    Class representing a field, a commutative ring for which all elements are units but 0.
    """

    def __getitem__(self, var):
        return PolynomialField(self, var)


class FiniteField(Field):

    # TODO generate base polynomial if possible
    def __init__(self, p, base_poly=None):
        super(FiniteField, self).__init__(Polynomial)
        if not IZ.is_prime(p):
            raise ValueError("p must be prime")
        if base_poly is None:
            self._base_ring = ModuloIntegers(p)
        else:
            self._base_ring = FiniteField(p)[base_poly.var]
            if not self._base_ring.is_irreducible(base_poly):
                raise ValueError(f"the polynomial must be irreducible in F({p})")
        self._p = p
        self._n = base_poly.degree if base_poly is not None else 1
        self._base_poly = base_poly if self._n != 1 else p

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
        return Polynomial([0], self._base_ring.var) if self._n != 1 else 0

    @property
    def one(self):
        return Polynomial([1], self._base_ring.var) if self._n != 1 else 1

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
        if self._n == 1:
            return self._base_ring.inverse(a)
        # a(t)x + q(t)y = 1 => a(t)x - 1 = (-y)q(t) => a(t)x ≡ 1 mod q(t)
        a = a @ self
        if a == self.zero:
            raise ZeroDivisionError
        gcd, (x, _) = self._base_ring.bezout(a, self._base_poly)
        return x @ self

    def discrete_logarithm(self, h, g):
        """TODO g ** x mod p(y) = h"""

        def get_s(k):
            def cond(x):
                if not isinstance(x, Polynomial):
                    if self._n == 1:
                        return x % 3 == k
                    return k == 0
                if x.degree < self._n - 1:
                    return k == 0
                return x.coefficients[-1] % 3 == k

            return cond

        def f(x, a, b):
            return (self.mul(h, x), a, Zn.add(b, Zn.one)) if s1(x) else \
                (self.pow(x, 2), Zn.times(a, 2), Zn.times(b, 2)) if s2(x) else \
                    (self.mul(g, x), Zn.add(a, Zn.one), b)

        h = h @ self
        g = g @ self

        if h == self.zero or g == self.zero:
            raise ValueError("cannot compute logarithm of 0 or with base 0")

        n = self.q - 1
        Zn = IZ(n)
        # chi = int(np.ceil(self.p / 3))
        # s3 = otherwise
        # s1, s2 = get_s(chi, 0), get_s(chi, 1)
        s1, s2 = get_s(1), get_s(0)
        x, a, b = self.one, Zn.zero, Zn.zero
        xs = [x, x]
        as_ = [a, a]
        bs = [b, b]
        while True:
            # it k -> x_k = g^{a_k} * h^{b_k}
            xs[0], as_[0], bs[0] = f(xs[0], as_[0], bs[0])
            x, a, b = f(xs[1], as_[1], bs[1])
            xs[1], as_[1], bs[1] = f(x, a, b)

            # x_k == x_{2k}
            if xs[0] == xs[1]:
                if self.sub(bs[0], bs[1]) == self.zero:
                    raise ValueError(f"cannot compute discrete_log({h}, {g})")
                break

        return self.mul(self.inverse(self.sub(as_[1], as_[0])), self.sub(bs[0], bs[1]))

    def eq(self, a, b):
        a = a @ self
        b = b @ self
        return a == b

    def contains(self, a):
        if isinstance(a, Polynomial) and isinstance(self._base_poly, Polynomial):
            return a.var == self._base_poly.var
        return isinstance(a, (int, np.integer)) or isinstance(a, (float, np.float)) and a.is_integer()

    def at(self, a):
        return self._base_ring.divmod(a, self._base_poly)[1] @ self._base_ring if self._n != 1 else a @ self._base_ring

    def __latex__(self):
        return fr"\mathbb{{F}}_{self.q}"


IF = FiniteField


class PolynomialField(Field):

    def __init__(self, base_ring, var):
        super(PolynomialField, self).__init__(Polynomial)
        self._base_ring = base_ring
        self._var = Var(var.x) if isinstance(var, Var) else Var(var)

    @property
    def var(self):
        return self._var

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
            coeffs = [0] * (k + 1)
            coeffs[1] = -1
            coeffs[-1] = 1
            return Polynomial(coeffs, self._var)

        p = p @ self
        if isinstance(self._base_ring, FiniteField):
            n = p.degree
            q = self._base_ring.q
            if self.divmod(poly(q ** n), p)[1] != 0:
                return False

            factors = set(IZ.factor(n))
            for r in factors:
                h = self.divmod(poly(q ** (n // r)), p)[1]
                if self.gcd(p, h) != 1:
                    return False

            return True

        raise ValueError(f"cannot check irreducibility in {self._base_ring}")

    def is_prime(self, p):
        return self.is_irreducible(p)

    def factor(self, a):
        pass

    def normal_part(self, a):
        a = a @ self
        if a == self.zero:
            return a
        lcoeff = a.coefficients[-1]
        return Polynomial([self._base_ring.div(ai, lcoeff) for ai in a.coefficients], self._var.x)

    def unit_part(self, a):
        a = a @ self
        if a == self.zero:
            return self.one
        return a.coefficients[-1]

    # GCD de los coeficientes
    def content(self, a):
        a @ self
        return self._base_ring.one

    # a = u(a) * cont(a) * pp(a)
    def primitive_part(self, a):
        a = a @ self
        return self.normal_part(a)

    # Divides a(x) / b(x) and returns it's quotient and remainder
    def divmod(self, a, b):
        """
        TODO
        """
        a = a @ self
        b = b @ self

        if a.var != b.var:
            raise ValueError("variables must be the same")

        quotient, remainder = self.zero, a

        while remainder.degree >= b.degree:
            monomial_exponent = remainder.degree - b.degree
            monomial_zeros = [self.zero for _ in range(monomial_exponent)]
            monomial_divisor = Polynomial(monomial_zeros +
                                          [self._base_ring.div(remainder.coefficients[-1], b.coefficients[-1])],
                                          a.var)

            quotient = self.add(quotient, monomial_divisor)
            remainder = self.sub(remainder, self.mul(monomial_divisor, b))

        return quotient, remainder

    def gcd(self, a, b, *args):
        c = self.primitive_part(a)
        d = self.primitive_part(b)

        while d != self.zero:
            r = self.divmod(c, d)[1]
            c = d
            d = self.primitive_part(r)

        g = c
        if args:
            return self.gcd(g, args[0], *args[1:])
        return g

    def bezout(self, a, b):
        """
        Returns two elements that satisfy Bezout's identity, that is, two elements x and y for which
        gcd(a, b) = a * x + b * y. It also returns the gcd of a and b.
        :param a: Polynomial - left-hand-side element
        :param b: Polynomial - right-hand-side element
        :return: (Polynomial, (Polynomial, Polynomial)) - gcd(a, b) and the two elements that satisfy Bezout's identity
        """
        r0 = self.normal_part(a)
        r1 = self.normal_part(b)

        x0 = self.one
        y0 = self.zero
        x1 = self.zero
        y1 = self.one
        while r1 != self.zero:
            (q, r2) = self.divmod(r0, r1)
            x2 = self.sub(x0, self.mul(q, x1))
            y2 = self.sub(y0, self.mul(q, y1))
            (x0, x1, y0, y1, r0, r1) = (x1, x2, y1, y2, r1, r2)
        x0 = self.divmod(x0, self.mul(self.unit_part(a), self.unit_part(r0)))[0]
        y0 = self.divmod(y0, self.mul(self.unit_part(b), self.unit_part(r0)))[0]
        return self.normal_part(r0), (x0, y0)

    def at(self, a):
        if a in self._base_ring:
            return Polynomial([a @ self._base_ring], self._var)
        if a not in self:
            raise ValueError("the element must be a polynomial")
        coeffs = a.coefficients
        return Polynomial([ai @ self._base_ring for ai in coeffs], self._var)

    def __latex__(self):
        return self._base_ring.__latex__() + "[" + self._var.__latex__() + "]"
