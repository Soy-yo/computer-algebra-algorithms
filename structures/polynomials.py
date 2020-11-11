from functools import reduce

import numpy as np


# TODO permitir operaciones para devolver polinomios (maybe)
class Var:

    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x

    def __add__(self, a):
        return Polynomial([a, 1], self.x)

    def __radd__(self, a):
        return self + a

    def __neg__(self):
        return Polynomial([0, -1], self.x)

    def __sub__(self, a):
        return self + (-a)

    def __rsub__(self, a):
        return self + (-a)

    def __mul__(self, a):
        return Polynomial([0, a], self.x)

    def __rmul__(self, a):
        return self * a

    def __pow__(self, n):
        if n < 0:
            return NotImplemented
        if n == 0:
            return 1
        result = [0] * (n + 1)
        result[-1] = 1
        return Polynomial(result, self.x)

    def as_polynomial(self):
        return Polynomial([0, 1], self.x)

    def __latex__(self):
        return self.x

    def _repr_latex_(self):
        return f"${self.__latex__()}$"

    def __repr__(self):
        return self.x


# TODO: función call p(5) y eso :)
class Polynomial:

    def __init__(self, coefficients, var, dtype=None):
        self._coefficients = np.trim_zeros(np.array(coefficients, dtype=dtype), trim='b')
        if self._coefficients.size == 0:
            self._coefficients = np.array([0], dtype=dtype)
        self._var = var if isinstance(var, Var) else Var(var)

    @property
    def coefficients(self):
        return self._coefficients.copy()

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, var):
        self._var = var if isinstance(var, Var) else Var(var)

    @property
    def degree(self):
        return len(self._coefficients) - 1

    def terms(self):
        return len([0 for t in self._coefficients if t != 0])

    def term(self, index):
        return self._coefficients[index]

    def __add__(self, q):
        if isinstance(q, Var):
            self._check_var(q)
            return self + q.as_polynomial()

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
        if isinstance(q, Var):
            self._check_var(q)
            return q.as_polynomial() + self

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
        if isinstance(q, Var):
            self._check_var(q)
            return q.as_polynomial() - self

        new_a0 = q - self._coefficients[0]
        if new_a0 == NotImplemented:
            return NotImplemented
        new_coeff = -self.coefficients
        new_coeff[0] = new_a0
        return Polynomial(new_coeff, self._var.x)

    # TODO: averiguar cómo no sacar floats
    def __mul__(self, q):
        if isinstance(q, Var):
            self._check_var(q)
            return self * q.as_polynomial()

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
        if isinstance(q, Var):
            self._check_var(q)
            return q.as_polynomial() * self

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

    def __eq__(self, q):
        if isinstance(q, Var):
            return self == q.as_polynomial()

        if not isinstance(q, Polynomial):
            if self.degree > 0:
                return False
            return self._coefficients[0] == q
        return self._var == q.var and (self._coefficients == q.coefficients).all()

    def _check_var(self, var):
        if self._var != var:
            raise ValueError(f"cannot operate with different variables: {self._var} and {var} found")

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
