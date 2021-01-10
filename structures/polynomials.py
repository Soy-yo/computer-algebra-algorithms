from functools import reduce
from itertools import accumulate

import numpy as np


class Polynomial:
    """
    Class representing a polynomial a_0 + a_1 x + ... + a_n x^n over a variable. In general, it uses numbers to
    represent its coefficients.
    """

    def __init__(self, coefficients, var):
        """
        :param coefficients: [any] - coefficients of the polynomial, starting with a0
        :param var: Var/str - variable of the polynomial
        """
        self._coefficients = Coefficients(coefficients)
        if isinstance(self, Var):
            self._var = self
        else:
            self._var = Var(var.x) if isinstance(var, Var) else Var(var)

    @property
    def coefficients(self):
        """
        Read-only view of the coefficients.
        :return: np.array - read-only view of the coefficients
        """
        return self._coefficients.copy()

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
        return sum(1 for t in self._coefficients if t != 0)

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
        coeffs = [0] * (n + 1)
        coeffs[n] = self._coefficients[n]
        return Polynomial(coeffs, self._var.x)

    def __add__(self, q):
        if not isinstance(q, Polynomial):
            return Polynomial(self._coefficients + q, self._var.x)
        elif q.var != self._var:
            if self.degree < 1:
                # n(x) + p(y) = p(y) + n
                return Polynomial(self._coefficients + q._coefficients, q.var.x)

            return Polynomial(self._coefficients + q, self._var.x)

        return Polynomial(self._coefficients + q._coefficients, self._var.x)

    def __radd__(self, q):
        return Polynomial(q + self._coefficients, self._var.x)

    def __neg__(self):
        return Polynomial(-self._coefficients, self._var.x)

    def __sub__(self, q):
        return self + (-q)

    def __rsub__(self, q):
        return q + (-self)

    def __mul__(self, q):
        if not isinstance(q, Polynomial):
            return Polynomial(self._coefficients * q, self._var)
        elif q.var != self._var:
            if self.degree < 1:
                # n(x) * p(y) = n * p(y)
                return Polynomial(self._coefficients * q._coefficients, q.var.x)
            return Polynomial(self._coefficients * q, self._var.x)

        return Polynomial(self._coefficients * q._coefficients, self._var.x)

    def __rmul__(self, q):
        return Polynomial(q * self._coefficients, self._var.x)

    def __pow__(self, n):
        if n < 0:
            raise ValueError("exponent must be positive or zero")
        if n == 0:
            return Polynomial([1], self._var.x)
        if n == 1:
            return Polynomial(self.coefficients, self._var.x)
        return reduce(lambda x, y: x * y, (self for _ in range(n)))

    def __eq__(self, q):
        if not isinstance(q, Polynomial):
            if self.degree > 0:
                return False
            return len(self._coefficients) == 0 and q == 0 or self._coefficients[0] == q
        if self.degree <= 0:
            return self.degree == q.degree and self._coefficients == q.coefficients
        return self._var == q.var and self.degree == q.degree and self._coefficients == q.coefficients

    def __call__(self, x):
        if not self._coefficients:
            return 0
        xs = accumulate([1] + [x] * self.degree, lambda a, b: a * b)
        return sum(a * x for a, x in zip(self._coefficients, xs))

    def derivative(self):
        """
        Return the derivative of this polynomial, that is, if f = sum(i=0, n; a_i*x^i), its derivative
        f' = sum(i=1, n; i * a_i * x^i-1).
        :return: Polynomial - the derivative of this polynomial
        """
        coeffs = [i * c for (i, c) in enumerate(self._coefficients)]
        return Polynomial(coeffs[1:], self._var.x)

    def _repr_coefficient(self, c, k, r):
        if c == 1 and k != 0:
            return ''
        if c == -1 and k != 0:
            return '-'
        if hasattr(c, '__len__') and len(c) > 1:
            return '(' + r(c) + ')'
        return r(c)

    def _repr_var(self, k, latex):
        if k == 0:
            return ''
        result = f'{self._var.__latex__()}' if latex else repr(self._var)
        if k == 1:
            return result
        return result + (f'^{{{k}}}' if latex else f'^{k}')

    def __latex__(self):
        if self.degree == -1:
            return '0'
        latex = getattr(self._coefficients, "__latex__", repr)
        return '+'.join([
            self._repr_coefficient(c, k, latex) + self._repr_var(k, latex=True)
            for k, c in reversed(list(enumerate(self._coefficients))) if c != 0
        ]).replace('+-', '-')

    def _repr_latex_(self):
        return f"${self.__latex__()}$"

    def __repr__(self):
        if self.degree == -1:
            return '0'
        return ' + '.join([
            self._repr_coefficient(c, k, repr) + self._repr_var(k, latex=False)
            for k, c in reversed(list(enumerate(self._coefficients))) if c != 0
        ]).replace('+ -', '- ')


class Var(Polynomial):
    """
    Class representing a variable. It extends Polynomial so it can be operated and create polynomials easily. For
    example:
    >>> x = Var('x')
    >>> f = x**2 + 2*x + 1 # returns the polynomial with coefficients [1, 2, 1]
    """

    x = None

    def __init__(self, x):
        """
        :param x: str - variable representation; accepts LaTeX formulas, for example '\lambda'
        """
        super().__init__([0, 1], self)
        self.x = x

    def __eq__(self, other):
        if not isinstance(other, Var):
            return False
        return self.x == other.x

    def __latex__(self):
        return self.x

    def __repr__(self):
        return self.x


class Coefficients:
    """
    Array-like class representing coefficients of a polynomial. It implements basic operators, such as addition,
    product, etc.
    """

    def __init__(self, a=None):
        """
        :param a: itreable - list of coefficients (default empty)
        """
        if a is None:
            a = []
        self._a = [c for c in a]
        self.trim()

    def __len__(self):
        return len(self._a)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            return self._a[item]
        if isinstance(item, slice):
            return Coefficients(self._a[item])

        return NotImplemented

    def __setitem__(self, key, value):
        if isinstance(key, (int, np.integer)):
            self._[key] = value
        elif isinstance(key, slice):
            for i, v in zip(range(key.start, key.stop, key.step), value):
                self._a[i] = v
        else:
            raise ValueError(f"undefined set method for type {type(key)}")

    def __neg__(self):
        return Coefficients([-x for x in self._a])

    def __add__(self, other):
        if not isinstance(other, Coefficients):
            return Coefficients([self._a[0] + other if self else other] + self._a[1:])

        a = self._a
        b = other._a
        if len(other) > len(self):
            a = a + [None] * (len(other) - len(self))
        elif len(self) > len(other):
            b = b + [None] * (len(self) - len(other))

        return Coefficients([ai + bi if ai is not None and bi is not None
                             else ai if ai is not None
        else bi for ai, bi in zip(a, b)])

    def __radd__(self, other):
        # isinstance(other, Coefficients) must be False
        return Coefficients([(other + self._a[0]) if self else other] + self._a[1:])

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if not isinstance(other, Coefficients):
            return Coefficients([x * other for x in self._a])

        if not self or not other:
            return Coefficients([])

        a = [0] * (len(self) + len(other) - 1)
        for i in range(len(self)):
            for j in range(len(other)):
                a[i + j] += self[i] * other[j]

        return Coefficients(a)

    def __rmul__(self, other):
        # isinstance(other, Coefficients) must be False
        return Coefficients([other * x for x in self._a])

    def __eq__(self, other):
        if not isinstance(other, Coefficients):
            return False
        return self._a == other._a

    def __bool__(self):
        return len(self) > 0

    def __repr__(self):
        return "Coefficients(" + repr(self._a) + ")"

    def __copy__(self):
        return self.copy()

    def copy(self):
        """
        Returns a copy of self.
        :return: Coefficients - a copy of this coefficients
        """
        return Coefficients(self._a)

    def trim(self):
        """
        Removes leading zeroes in this coefficients.
        """
        while self._a and self._a[-1] == 0:
            self._a.pop()
