import abc
from functools import reduce
from itertools import accumulate

import numpy as np

from . import domains, fields, rings


class Polynomial:
    """
    Class representing a polynomial a_0 + a_1 x + ... + a_n x^n over a variable. In general, it uses numbers to
    represent its coefficients.
    """

    def __init__(self, coefficients, var, dtype=None):
        """
        :param coefficients: [any] - coefficients of the polynomial, starting with a0
        :param var: Var/str - variable of the polynomial
        :param dtype: type - dtype to pass to numpy array
        """
        self._coefficients = np.trim_zeros(np.array(coefficients, dtype=dtype), trim='b')
        if self._coefficients.size == 0:
            self._coefficients = np.array([0], dtype=dtype)
        self._var = Var(var.x) if isinstance(var, Var) else Var(var)

    @property
    def coefficients(self):
        """
        Read-only view of the coefficients.
        :return: np.array - read-only view of the coefficients
        """
        coeff = self._coefficients.view()
        coeff.flags.writeable = False
        return coeff

    @property
    def var(self):
        """
        Variable instance of this polynomial.
        :return: Var - variable instance of this polynomial
        """
        return self._var

    @var.setter
    def var(self, var):
        """
        Sets the variable of this polynomial.
        :param var: Var/str - variable of the polynomial
        """
        self._var = Var(var.x) if isinstance(var, Var) else Var(var)

    @property
    def degree(self):
        """
        Degree of this polynomial, that is, the maximum exponent of the variable in all terms.
        :return: int - degree of this polynomial
        """
        return len(self._coefficients) - 1

    @property
    def terms(self):
        """
        Number of terms in the polynomial.
        :return: int - number of terms in the polynomial
        """
        return len([0 for t in self._coefficients if t != 0])

    def __len__(self):
        return self.terms

    def term(self, n):
        """
        Returns the term of degree n.
        :param n: int - degree of the term
        :return: Polynomial - polynomial representing the n-th term
        """
        if n > self.degree:
            raise ValueError(f"{self} has degree lower than {n}")
        coeffs = self._coefficients.copy()[:n]
        coeffs[:n - 1] = 0
        return Polynomial(coeffs, self._var.x, self._coefficients.dtype)

    def __add__(self, q):
        if not isinstance(q, Polynomial):
            new_a0 = self._coefficients[0] + q
            if new_a0 == NotImplemented:
                return NotImplemented
            new_coeff = self.coefficients
            new_coeff[0] = new_a0
            return Polynomial(new_coeff, self._var.x)

        self._check_var(q.var)
        coeff_p = self._coefficients
        coeff_q = q._coefficients
        if self.degree > q.degree:
            coeff_q = np.pad(coeff_q, (0, self.degree - q.degree))
        elif q.degree > self.degree:
            coeff_p = np.pad(coeff_p, (0, q.degree - self.degree))
        return Polynomial(coeff_p + coeff_q, self._var.x)

    def __radd__(self, q):
        new_a0 = q + self._coefficients[0]
        if new_a0 == NotImplemented:
            return NotImplemented
        new_coeff = self.coefficients
        new_coeff[0] = new_a0
        return Polynomial(new_coeff, self._var.x)

    def __neg__(self):
        return Polynomial(-self._coefficients, self._var.x)

    def __sub__(self, q):
        return self + (-q)

    def __rsub__(self, q):
        new_a0 = q - self._coefficients[0]
        if new_a0 == NotImplemented:
            return NotImplemented
        new_coeff = -self.coefficients
        new_coeff[0] = new_a0
        return Polynomial(new_coeff, self._var.x)

    # TODO: averiguar cómo no sacar floats
    def __mul__(self, q):
        if not isinstance(q, Polynomial):
            new_coeff = self._coefficients * q
            if new_coeff[0] == NotImplemented:
                return NotImplemented
            return Polynomial(new_coeff, self._var.x)

        self._check_var(q.var)
        result = np.zeros(self.degree + q.degree + 1)
        for i in range(self.degree + 1):
            for j in range(q.degree + 1):
                result[i + j] += self._coefficients[i] * q._coefficients[j]
        return Polynomial(result, self._var.x)

    def __rmul__(self, q):
        new_coeff = q * self._coefficients
        if new_coeff[0] == NotImplemented:
            return NotImplemented
        return Polynomial(new_coeff, self._var.x)

    def __pow__(self, n):
        if n <= 0:
            return NotImplemented
        if n == 1:
            return self
        return reduce(lambda x, y: x * y, (self for _ in range(n)))

    def __xor__(self, n):
        return self.__pow__(n)

    def __eq__(self, q):
        if not isinstance(q, Polynomial):
            if self.degree > 0:
                return False
            return self._coefficients[0] == q
        return self._var == q.var and (self._coefficients == q.coefficients).all()

    def __call__(self, x):
        xs = np.array(list(accumulate([1] + [x] * self.degree, lambda a, b: a * b)))
        return (self._coefficients * xs).sum()

    def derivative(self):
        """
        Return the derivative of this polynomial, that is, if f = sum(i=0, n; a_i*x^i), its derivative
        f' = sum(i=1, n; i * a_i * x^i-1).
        :return: Polynomial - the derivative of this polynomial
        """
        coeffs = [i * c for (i, c) in enumerate(self._coefficients)]
        return Polynomial(coeffs[1:], self._var.x, self._coefficients.dtype)

    def _check_var(self, var):
        if self._var != var:
            raise ValueError(f"cannot operate with different variables: {self._var} and {var} found")

    # TODO add parentheses if a_i has length and len(a_i) > 1
    def __latex__(self):
        latex = getattr(self._coefficients, "__latex__", None)
        return '+'.join([
            ((latex(c) if latex is not None else str(c)) if c != 1 or k == 0 else '') +
            (f'{self._var.__latex__()}' if k != 0 else '') +
            (f'^{{{k}}}' if k not in (0, 1) else '')
            for k, c in reversed(list(enumerate(self._coefficients)))
            if c != 0
        ]).replace('+-', '-')

    def _repr_latex_(self):
        return f"${self.__latex__()}$"

    def __repr__(self):
        return ' + '.join([
            (str(c) if c != 1 or k == 0 else '') +
            (str(self._var) if k != 0 else '') +
            (f'^{k}' if k not in (0, 1) else '')
            for k, c in reversed(list(enumerate(self._coefficients)))
            if c != 0
        ]).replace('+ -', '- ')


class Var(Polynomial):
    """
    Class representing a variable. It extends Polynomial so it can be operated and create polynomials easily. For
    example:
    >>> x = Var('x')
    >>> f = x**2 + 2*x + 1 # returns the polynomial with coefficients [1, 2, 1]
    """

    def __init__(self, x, dtype=None):
        super().__init__([0, 1], self, dtype)
        self.x = x

    def __latex__(self):
        return self.x

    def __repr__(self):
        return self.x


class PolynomialRing(rings.UnitaryRing, abc.ABC):

    def __init__(self, base_ring, var):
        super(PolynomialRing, self).__init__(Polynomial)
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

    # TODO ???
    def is_unit(self, a):
        pass

    def inverse(self, a):
        pass

    def is_zero_divisor(self, a):
        pass

    def contains(self, a):
        return a in self._base_ring or all(ai in self._base_ring for ai in a.coefficients)

    def at(self, a):
        if a in self._base_ring:
            return Polynomial([a @ self._base_ring], self._var)
        if a not in self:
            raise ValueError("the element must be a polynomial")
        coeffs = a.coefficients
        return Polynomial([ai @ self._base_ring for ai in coeffs], self._var, coeffs.dtype)

    def __latex__(self):
        return self._base_ring.__latex__() + "[" + self._var.__latex__() + "]"


class PolynomialUFD(domains.UFD, PolynomialRing):

    def __init__(self, base_ring, var):
        super(PolynomialUFD, self).__init__(base_ring, var)

    def is_prime(self, p):
        pass

    def factor(self, a):
        pass

    def normal_part(self, a):
        a = a @ self
        u = self.unit_part(a)
        return self.divmod(a, u)[0]

    def unit_part(self, a):
        a = a @ self
        return self._base_ring.unit_part(a.coefficients[-1])

    def is_unit(self, a):
        pass

    def inverse(self, a):
        pass

    # GCD de los coeficientes
    def content(self, a):
        a = a @ self
        return self._base_ring.gcd(*a.coefficients)

    # a = u(a) * cont(a) * pp(a)
    def primitive_part(self, a):
        a = a @ self
        return self.divmod(a, self.mul(self.unit_part(a), self.content(a)))[0]

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


class PolynomialField(fields.Field, PolynomialUFD):

    def __init__(self, base_ring, var):
        super(PolynomialField, self).__init__(base_ring, var)
