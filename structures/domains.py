import abc

from structures.polynomials import Polynomial, Var
from structures.rings import Ring, CommutativeRing, UnitaryRing


class Domain(Ring, abc.ABC):
    """
    Class representing a domain. That is, a ring in which 0 is the only zero divisor.
    """

    @property
    def is_domain(self):
        return True

    def is_zero_divisor(self, a):
        return a == self.zero


class IntegralDomain(CommutativeRing, Domain, abc.ABC):
    """
    Class representing an integral domain. That is, a commutative ring in which 0 is the only zero divisor.
    """

    def normal_part(self, a):
        """
        Returns the normal part of the given element a. The unit normal element of an unit normal class
        [a] := {b: a = bu, is_unit(u)} is the canonical representative of that class. The normal part of a satisfies
        a = n(a) * u(a).
        :param a: self.dtype - element for which compute its normal part
        :return: self.dtype - normal part of a
        """
        raise NotImplementedError

    def unit_part(self, a):
        """
        Returns the unit part of the given element a. The unit normal element of an unit normal class
        [a] := {b: a = bu, is_unit(u)} is the canonical representative of that class. The unit part of a satisfies
        a = n(a) * u(a).
        :param a: self.dtype - element for which compute its unit part
        :return: self.dtype - unit part of a
        """
        raise NotImplementedError


class UFD(IntegralDomain, UnitaryRing, abc.ABC):
    """
    Class representing of an Unique Factorization Domain, an integral domain where all elements can be written as a
    product of prime elements (or irreducible elements), uniquely up to order and units.
    """

    def is_prime(self, p):
        """
        Determines whether the element p is prime or not. A prime element is a non zero element for which the principal
        ideal (p) generated by p is a nonzero prime ideal.
        :param p: self.dtype - element to be checked
        :return: bool - True if p is prime, False otherwise
        """
        raise NotImplementedError

    def factor(self, a):
        """
        Returns a list containing the unique prime factorization of a. If an element has multiplicity n,
        it will appear n times, and 1 won't be in the list.
        :param a: self.dtype - element to be factorized
        :return: [self.dtype] - list of prime elements that decompose a
        """
        raise NotImplementedError

    def __getitem__(self, var):
        return PolynomialUFD(self, var)


class EuclideanDomain(UFD, abc.ABC):
    """
    Class representing an euclidean domain, a domain endowed with a Euclidean function which allows a suitable
    generalization of the Euclidean division of the integers. In any Euclidean domain, one can apply the Euclidean
    algorithm to compute the greatest common divisor of any two elements.
    """

    def value(self, a):
        """
        Computes the Euclidean function over a. A Euclidean function v : R\\{0} -> IZ+ must satisfy the following:
        - if b != 0 and there exist q, r such that a = b*q + r, then r = 0 or v(r) < v(b)
        - if b != 0, v(a) <= v(a*b)
        This function will vary depending on the ring.
        :param a: self.dtype - element for which compute the Euclidean function
        :return: int - result of applying the Euclidean function to a
        """
        raise NotImplementedError

    def quot(self, a, b):
        """
        Returns the quotient q of the division a = b * q + r.
        :param a: self.dtype - dividend
        :param b: self.dtype - divisor
        :return: self.dtype - quotient of the division
        """
        return self.divmod(a, b)[0]

    def rem(self, a, b):
        """
        Returns the remainder r of the division a = b * q + r.
        :param a: self.dtype - dividend
        :param b: self.dtype - divisor
        :return: self.dtype - remainder of the division
        """
        return self.divmod(a, b)[1]

    def divmod(self, a, b):
        """
        Returns both the quotient q and the remainder r of the division a = b * q + r.
        :param a: self.dtype - dividend
        :param b: self.dtype - divisor
        :return: (self.dtype, self.dtype) - quotient and remainder of the division
        """
        raise NotImplementedError

    def divides(self, a, b):
        """
        Determines whether a divides b or not (a|b), that is if there exists an element c for which b = a * c.
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side
        :return: bool - True if a|b, else False
        """
        return self.rem(a, b) == self.zero

    def gcd(self, a, b, *args):
        """
        Returns the greatest common divisor of a and b, that is, the greatest element which divides both a and b.
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side element
        :param args: self.dtype - undefined number of extra elements to compute gcd for
        :return: self.dtype - greatest common divisor of a and b
        """
        r0 = self.normal_part(a)
        r1 = self.normal_part(b)
        while r1 != self.zero:
            r2 = self.rem(r0, r1)
            r0 = r1
            r1 = r2
        if args:
            return self.gcd(self.normal_part(r0), args[0], *args[1:])
        return self.normal_part(r0)

    def lcm(self, a, b):
        """
        Returns the least common multiple of a and b, that is, the lowest element which is divided by both a and b.
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side element
        :return: self.dtype - least common multiple of a and b
        """
        return self.quot(self.normal_part(self.mul(a, b)), self.gcd(a, b))

    def bezout(self, a, b):
        """
        Returns two elements that satisfy Bezout's identity, that is, two elements x and y for which
        gcd(a, b) = a * x + b * y. It also returns the gcd of a and b.
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side element
        :return: (self.dtype, (self.dtype, self.dtype)) - gcd(a, b) and the two elements that satisfy Bezout's identity
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
        x0 = self.quot(x0, self.mul(self.unit_part(a), self.unit_part(r0)))
        y0 = self.quot(y0, self.mul(self.unit_part(b), self.unit_part(r0)))
        return self.normal_part(r0), (x0, y0)

    def are_coprime(self, a, b):
        """
        Returns whether a and b are coprime or not, that is, gcd(a, b) == 1.
        :param a: self.dtype - one element to check coprimality
        :param b: self.dtype - another element to check coprimality
        :return: bool - True if a and b are coprime, False otherwise
        """
        return self.gcd(a, b) == self.one


# TODO sobreescribir __div__ para poder hacer IZ_p[x]/(f(x))
class PolynomialUFD(UFD):
    """
    Class representing D[x], an UFD of polynomials over a domain D with variable x.
    """

    def __init__(self, base_domain, var):
        """
        :param base_domain: Domain - domain where coefficients live in
        :param var: Var or str - variable of polynomials in this UFD
        """
        super(PolynomialUFD, self).__init__(Polynomial)
        self._base_ring = base_domain
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

    # TODO ???
    def is_unit(self, a):
        pass

    def inverse(self, a):
        pass

    def is_zero_divisor(self, a):
        pass

    def is_irreducible(self, p):
        pass

    def is_prime(self, p):
        return self.is_irreducible(p)

    def factor(self, a):
        pass

    def normal_part(self, a):
        a = a @ self
        u = self.unit_part(a)
        return self.divmod(a, u, pseudo=False)[0]

    def unit_part(self, a):
        a = a @ self
        if a == self.zero:
            return self.one
        return self._base_ring.unit_part(a.coefficients[-1])

    def content(self, a):
        """
        Returns the gcd of the coefficients of a.
        :param a: Polynomial - polynomial for which compute its content
        :return: self.base_ring.dtype - gcd of the polynomial's coefficients
        """
        a = a @ self
        if a.degree < 1:
            return a.coefficients[0] if a.degree == 0 else 0
        return self._base_ring.gcd(*a.coefficients)

    # a = u(a) * cont(a) * pp(a)
    def primitive_part(self, a):
        """
        Returns the primitive part of the given polynomial, that is, the polynomial divided by its primitive part
        (unit-normalized).
        :param a: Polynomial - polynomial for which compute its primitive part
        :return: Polynomial - primitive part of a
        """
        a = a @ self
        if a == self.zero:
            return self.zero
        return self.divmod(a, self.mul(self.unit_part(a), self.content(a)), pseudo=False)[0]

    def divmod(self, a, b, pseudo=True):
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

        if b == self.zero:
            raise ZeroDivisionError

        if a.degree < b.degree:
            return self.zero, a

        if pseudo:
            beta = b.coefficients[-1]
            ell = a.degree - b.degree + 1
            a = self.mul(a, beta ** ell)

        quotient, remainder = self.zero, a

        while remainder.degree >= b.degree:
            monomial_exponent = remainder.degree - b.degree
            monomial_zeros = [self.zero for _ in range(monomial_exponent)]
            monomial_divisor = Polynomial(monomial_zeros +
                                          [self._base_ring.quot(remainder.coefficients[-1], b.coefficients[-1])],
                                          a.var)

            quotient = self.add(quotient, monomial_divisor)
            remainder = self.sub(remainder, self.mul(monomial_divisor, b))

        return quotient, remainder

    # TODO sacar fuera las funciones repetidas
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

    def contains(self, a):
        return a in self._base_ring or all(ai in self._base_ring for ai in a.coefficients)

    def __eq__(self, other):
        return isinstance(other, PolynomialUFD) and self._var == other._var and self._base_ring == other._base_ring

    def at(self, a):
        if a in self._base_ring:
            return Polynomial([a @ self._base_ring], self._var)
        if a not in self:
            raise ValueError("the element must be a polynomial")
        coeffs = a.coefficients
        return Polynomial([ai @ self._base_ring for ai in coeffs], self._var)

    def __latex__(self):
        return self._base_ring.__latex__() + "[" + self._var.__latex__() + "]"
