import abc
import itertools as it
import random

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
    """
    Class representing a finite field F_q = Z_p[x]/(f(x)), such that q = p^n for p prime and f a polynomial of degree n.
    If n = 1 this is isomorphic to Z_p but this implementation inherits all Field methods instead of just
    UnitaryRing ones.
    Its elements are the equivalence classes of polynomials over Z_p[x] modulo f(x) and their representatives are
    taken as the ones with degree lower than n.
    """

    # TODO generate base polynomial if possible
    def __init__(self, p, base_poly=None):
        """
        :param p: int - prime number, desired modulo Z_p
        :param base_poly: Polynomial or None - if None (default) this field is Z_p; otherwise it is
                          Z_p[x]/(base_poly(x))
        """
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
    def char(self):
        return self._p

    @property
    def elements(self):
        if self._n == 1:
            return list(range(self._p))
        return [Polynomial(cs, self._base_ring.var) for cs in it.product(*list(it.repeat(range(self._p), self._n)))]

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

    def pow(self, a, n):
        a = a @ self
        if isinstance(a, Polynomial):
            return (a ** n) @ self
        return pow(a, n, mod=self.p)

    def inverse(self, a):
        if self._n == 1:
            return self._base_ring.inverse(a)
        # a(t)x + q(t)y = 1 => a(t)x - 1 = (-y)q(t) => a(t)x ≡ 1 mod q(t)
        a = a @ self
        if a == self.zero:
            raise ZeroDivisionError
        gcd, (x, _) = self._base_ring.bezout(a, self._base_poly)
        return x @ self

    def discrete_logarithm(self, g, h):
        """
        Returns the discrete logarithm of h with base g, that is, an element x such that g ** x = h mod P. If this
        finite field is Fp such that p is prime, then g must be a generator of the multiplicative group of Z_{(p-1)/2}.
        Otherwise, g must be a generator of the multiplicative group of F_q such that q = p^n.
        This algorithm may fail with a low probability, inverse to self.q or if the previous condition is not
        fulfilled. In these cases None will be returned.
        :param g: int or Polynomial - base of the logarithm
        :param h: int or Polynomial - argument of the logarithm
        :return: x: int - element that satisfies g ** x = h mod P
        """

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
            return (self.mul(x, g), Zn.add(a, Zn.one), b) if s0(x) else \
                (self.mul(x, h), a, Zn.add(b, Zn.one)) if s1(x) else \
                    (self.pow(x, 2), Zn.times(a, 2), Zn.times(b, 2))

        h = h @ self
        g = g @ self

        if h == self.zero or g == self.zero:
            raise ValueError("cannot compute logarithm of 0 or with base 0")

        # TODO check if g is a generator of whatever group it should be
        n = self.q - 1
        if self._n == 1:
            n //= 2
        Zn = IZ(n)
        # s2 = otherwise
        s0, s1 = get_s(0), get_s(1)
        x, a, b = self.one, Zn.zero, Zn.zero
        x_, a_, b_ = x, a, b
        while True:
            # it k -> x_k = g^{a_k} * h^{b_k}
            x, a, b = f(x, a, b)
            x_, a_, b_ = f(*f(x_, a_, b_))

            # x_k == x_{2k}
            if x == x_:
                break

        r = Zn.sub(b, b_)
        s = Zn.sub(a_, a)
        if not Zn.is_unit(r):
            # Couldn't compute
            return None
        x = Zn.mul(Zn.inverse(r), s)
        if self.eq(self.pow(g, x), h):
            return x
        return self.add(x, n)

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

    def __eq__(self, other):
        return isinstance(other, FiniteField) and self.q == other.q and self._base_poly == other._base_poly

    def __latex__(self):
        return fr"\mathbb{{F}}_{self.q}"


IF = FiniteField


class PolynomialField(Field):
    """
    Class representing K[x], a field of polynomials over a field K with variable x.
    """

    def __init__(self, base_field, var):
        """
        :param base_field: Field - field where coefficients live in
        :param var: Var or str - variable of polynomials in this field
        """
        super(PolynomialField, self).__init__(Polynomial)
        self._base_ring = base_field
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
        return (a + b) @ self

    def mul(self, a, b):
        a = a @ self
        b = b @ self
        return (a * b) @ self

    def negate(self, a):
        a = a @ self
        return (-a) @ self

    def pow(self, a, n):
        a = a @ self
        return (a ** n) @ self

    def eq(self, a, b):
        a = a @ self
        b = b @ self
        return a == b

    def is_unit(self, a):
        pass

    def inverse(self, a):
        pass

    def is_zero_divisor(self, a):
        pass

    def contains(self, a):
        return a in self._base_ring or all(ai in self._base_ring for ai in a.coefficients)

    def is_irreducible(self, p):
        """
        Determines whether the given polynomial is irreducible or not, that is, it cannot be expressed as P = f*g
        for any other polynomials f and g. A more suitable alias for is_prime.
        Valid for FiniteFields only as they are the only fields implemented for the moment.
        This algorithm becomes really slow when the base ring is Fq with q = p^n, n > 1 and deg(P) > 2.
        :param p: Polynomial over FiniteField - polynomial to determine if it is irreducible or not
        :return: bool - True if p is irreducible, False otherwise
        """

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

    def factor(self, f, method=None):
        """
        Returns the factorization of the given polynomial f, that is, a set of irreducible polynomials f1, ...,
        fk such that f = f1 * ... * fk. Implemented only for polynomials over finite fields.
        There exists three algorithms for polynomials over finite fields: Berlekamp Factorization Algorithm,
        Cantor/Zassenhaus and three-step factor and all are implemented here.
        BFA works well only when q of the finite field is relatively small.
        Cantor/Zassenhaus algorithm is a probabilistic enhanced version of Berlekamp, capable of working with
        larger primes.
        Three-step algorithm uses square-free factorization, distinct-degree factorization and equal-degree
        factorization.
        :param f: Polynomial over FiniteField - polynomial to be factored
        :param method: str - algorithm to be used; one of 'bfa' (Berlekamp Factorization Algorithm, default), 'cz'
                             (Cantor/Zassenhaus) or 'ts' (three-step)
        :return: [Polynomial] - list of factors of f
        """
        # Make the polynomial monic
        f = f @ self
        c0 = f.coefficients[-1]
        f = self.divmod(f, c0)[0]
        if f.degree == 1:
            return [f] if c0 == self.one else [f, c0]

        if isinstance(self._base_ring, FiniteField):
            if method is None:
                method = 'bfa'

            if method == 'bfa' or method == 'cz':
                def bfa(f):
                    phi = self._construct_phi(f)
                    for i in range(len(phi)):
                        phi[i][i] = self._base_ring.sub(phi[i][i], self._base_ring.one)
                    vs = self._ker(phi)
                    vs = [Polynomial(v, f.var) @ self for v in vs]
                    factors = {f}
                    r = 1
                    while len(factors) < len(vs):
                        facs = set()
                        for u in factors:
                            # Berlekamp splitting
                            for s in self._base_ring.elements:
                                g = self.gcd(self.sub(vs[r], s), u)
                                if g != self.one or g != u:
                                    u = self.divmod(u, g)[0]
                                    facs |= {u, g}
                                if len(facs) == len(vs):
                                    return list(facs)
                        factors = facs
                        r += 1

                    return list(factors)

                def cz(f):
                    def random_linear():
                        n = self._base_ring.n
                        if n == 1:
                            return tuple(np.random.randint(1, self._base_ring.p, s, dtype=int))
                        coeffs = np.random.randint(0, self._base_ring.p, (s, n), dtype=int)
                        for i in range(s):
                            while (coeffs[i] == 0).all():
                                coeffs[i] = np.random.randint(0, self._base_ring.p, n, dtype=int)
                        return tuple(Polynomial(c, self._var) for c in coeffs)

                    phi = self._construct_phi(f)
                    for i in range(len(phi)):
                        phi[i][i] = self._base_ring.sub(phi[i][i], self._base_ring.one)
                    vs = self._ker(phi)
                    vs = [Polynomial(v, f.var) @ self for v in vs]

                    factors = {f}
                    s = len(vs)
                    while len(factors) < s:
                        g = random.choice(tuple(h for h in factors if h.degree > 1))
                        cs = random_linear()
                        h = sum(c * v for c, v in zip(cs, vs))
                        # q must be odd
                        w = self.gcd(g, h ** ((self._base_ring.q - 1) // 2) - 1)
                        if w not in (1, g):
                            factors = (factors - {g}) | {w, self.divmod(g, w)[0]}

                    return list(factors)

                factor_fun = bfa if method == 'bfa' else cz

                factors = []
                polys = self.square_free_factorization(f)
                for g, k in polys:
                    result = factor_fun(g)
                    for _ in range(k):
                        # TODO maybe is not a good a idea not copying polynomials
                        factors.extend(result[:])

                if c0 != self.one:
                    factors.append(c0)

                # Dirty fix as it was returning some ones
                return [a for a in factors if a != self.one]

            if method == 'ts':
                factors = []
                polys = self.square_free_factorization(f)
                for g, k in polys:
                    polys_ = self.distinct_degree_factorization(g)
                    for h, j in polys_:
                        result = self.equal_degree_factorization(h, j)
                        for _ in range(k):
                            factors.extend(result[:])

                if c0 != self.one:
                    factors.append(c0)

                return factors

        raise ValueError(f"cannot factor in {self._base_ring}")

    def square_free_factorization(self, f):
        """
        Returns the square-free factorization of the given monic polynomial f, that is, coprime square-free
        polynomials f1, ..., fk such that f = f1 * ... * fk^k. Implemented only for polynomials over finite fields.
        :param f: Polynomial over FiniteField - polynomial to be factored
        :return: [(Polynomial, int)] - list of pairs factor and its exponent
        """
        if isinstance(self._base_ring, FiniteField):
            def pth_root(g):
                k = p ** (n - 1)
                coeffs = [self._base_ring.pow(c, k) for c in g.coefficients[::p]]
                return Polynomial(coeffs, g.var)

            f = f @ self
            p = self._base_ring.p
            n = self._base_ring.n

            result = []
            s = 1
            while f != self.one:
                j = 1
                g = self.divmod(f, self.gcd(f, f.derivative()))[0]
                while g != self.one:
                    f = self.divmod(f, g)[0]
                    h = self.gcd(f, g)
                    m = self.divmod(g, h)[0]
                    if m != self.one:
                        result.append((m, j * s))
                    g = h
                    j += 1
                if f != self.one:
                    f = pth_root(f)
                    s = p * s

            return result

        raise ValueError(f"cannot square-free factor in {self._base_ring}")

    def distinct_degree_factorization(self, f):
        """
        Returns the distinct-degree factorization of the given monic square-free polynomial f, that is,
        a set of polynomials f_{i_1}, ..., f_{i_k}, where each f_{i_j} is a product of polynomials of degree i_j, for
        j in {1, ..., k} such that f = f_{i_1} * ... * f_{i_k}. Implemented only for polynomials over finite fields.
        :param f: Polynomial over FiniteField - polynomial to be factored
        :return: [(Polynomial, int)] - list of pairs factor and its degree
        """
        if isinstance(self._base_ring, FiniteField):
            f = f @ self
            result = []
            h = f.var
            k = 0
            while f != self.one:
                h = self.divmod(self.pow(h, self._base_ring.q), f)[1]
                k += 1
                g = self.gcd(h - f.var, f)
                if g != self.one:
                    result.append((g, k))
                    f = self.divmod(f, g)[0]
                    h = self.divmod(h, f)[1]
            return result

        raise ValueError(f"cannot distinct-degree factor in {self._base_ring}")

    def equal_degree_factorization(self, f, k):
        """
        Returns the equal-degree factorization of the given monic square-free polynomial f, that is,
        a set of irreducible polynomials f_1, ..., f_r of degree k that they factorize f (f = f_1 * ... * f_r).
        Implemented only for polynomials over finite fields.
        :param f: Polynomial over FiniteField - polynomial to be factored
        :param k: int - expected degree of the returned polynomials (must divide degree of f)
        :return: [Polynomial] - list of polynomials of degree k that factorize f
        """
        if f.degree % k != 0:
            raise ValueError(f"k must divide deg(f), but {k}∤{f.degree}")

        if isinstance(self._base_ring, FiniteField):
            def mk(a):
                x = f.var
                poly = sum(x ** (2 ** j) for j in range(self._base_ring.n * k)) @ self
                return poly(a) @ self

            def random(n):
                m = self._base_ring.n * n
                return Polynomial(np.random.randint(0, self._base_ring.p, m, dtype=int), f.var)

            f = f @ self
            result = [f]
            r = f.degree // k
            while len(result) < r:
                # Original code
                factors = []
                for h in result:
                    a = random(h.degree)
                    d = self.gcd(mk(a), h)
                    if d == self.one or d == h:
                        factors.append(h)
                    else:
                        factors.append(d)
                        factors.append(self.divmod(h, d)[0])
                result = factors

            return result

        raise ValueError(f"cannot distinct-degree factor in {self._base_ring}")

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

    def content(self, a):
        """
        Returns the gcd of the coefficients of a. As this is a field, just 1.
        :param a: Polynomial - polynomial for which compute its content
        :return: self.base_ring.dtype - self.base_ring.one
        """
        a @ self
        return self._base_ring.one

    # a = u(a) * cont(a) * pp(a)
    def primitive_part(self, a):
        """
        Returns the primitive part of the given polynomial, that is, the polynomial divided by its primitive part
        (unit-normalized). As this is a field, the primitive part of a is just its normal part (content(a) = 1).
        :param a: Polynomial - polynomial for which compute its primitive part
        :return: Polynomial - primitive part of a
        """
        a = a @ self
        return self.normal_part(a)

    def divmod(self, a, b):
        """
        Returns the quotient and remainder of the division between polynomials a and b.
        :param a: Polynomial - dividend
        :param b: Polynomial - divisor
        :return: (Polynomial, Polynomial) - quotient and remainder of a/b
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

    def __eq__(self, other):
        return isinstance(other, PolynomialField) and self._var == other._var and self._base_ring == other._base_ring

    def __latex__(self):
        return self._base_ring.__latex__() + "[" + self._var.__latex__() + "]"

    def _construct_phi(self, f):
        q = self._base_ring.q
        n = f.degree
        phi = [None] * n
        r = [self.one] + [self.zero] * (n - 1)  # <-- n
        phi[0] = r
        coeffs = f.coefficients[:-1]  # <-- n
        F = self._base_ring
        for i in range(1, (n - 1) * q + 1):
            r = [F.sub(ri, F.mul(r[-1], c)) for c, ri in zip(coeffs, [self._base_ring.zero] + r[:-1])]
            if i % q == 0:
                phi[i // q] = r

        return phi

    def _ker(self, m):
        n = len(m)
        F = self._base_ring
        for k in range(n):
            i = k
            while i < n and m[k][i] == F.zero:
                i += 1
            if i < n:
                pivot = m[k][i]
                inv = F.inverse(pivot)
                # Normalization
                for j in range(n):
                    m[j][i] = F.mul(m[j][i], inv)
                # Swap columns i and k
                if i != k:
                    for j in range(n):
                        m[j][i], m[j][k] = m[j][k], m[j][i]
                # Make zeros
                for i in range(n):
                    if i == k:
                        continue
                    mki = m[k][i]
                    for j in range(n):
                        m[j][i] = F.sub(m[j][i], F.mul(m[j][k], mki))

        for i in range(n):
            m[i][i] = F.sub(m[i][i], F.one)

        j = 0
        vs = []
        while j < n:
            while j < n and all(a == F.zero for a in m[j]):
                j += 1
            if j < n:
                vs.append(m[j])
                j += 1

        return vs
