import numpy as np


class Var:

    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x

    def __latex__(self):
        return self.x

    def _repr_latex_(self):
        return f"${self.__latex__()}$"

    def __repr__(self):
        return self.x


class Polynomial:

    def __init__(self, coefficients, var, ring):
        self._coefficients = np.trim_zeros(np.array(coefficients), trim='b')
        if self._coefficients.size == 0:
            self._coefficients = np.array([ring.zero])
        self._var = var if isinstance(var, Var) else Var(var)
        self._ring = ring

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
    def ring(self):
        return self._ring

    @property
    def degree(self):
        return len(self._coefficients) - 1

    # TODO allow self + element of ring (or element of ring + self)
    def __add__(self, q):
        if not isinstance(q, Polynomial):
            return NotImplemented

        self._check_ring(q.ring)
        self._check_var(q.var)
        coeff_p = self._coefficients
        coeff_q = q._coefficients
        if self.degree > q.degree:
            coeff_q = np.pad(coeff_q, (0, q.degree - self.degree))
        elif q.degree > self.degree:
            coeff_p = np.pad(coeff_p, (0, self.degree - q.degree))
        return Polynomial(coeff_p + coeff_q, self._var.x, self._ring)

    def __neg__(self):
        return Polynomial(-self._coefficients, self._var.x, self._ring)

    def __sub__(self, q):
        if not isinstance(q, Polynomial):
            return NotImplemented

        return self + (-q)

    def __mul__(self, q):
        # TODO
        pass

    def __pow__(self, n):
        # TODO
        pass

    def __eq__(self, q):
        return self._ring == q.ring and self._var == q.var and (self._coefficients == q.coefficients).all()

    # def __truediv__(self, q):
    #     pass

    def _check_ring(self, ring):
        if self._ring != ring:
            raise ValueError(f"cannot operate with different rings: {self._ring} and {ring} found")

    def _check_var(self, var):
        if self._var != var:
            raise ValueError(f"cannot operate with different variables: {self._var} and {var} found")

    def __latex__(self):
        latex = getattr(self._coefficients, "__latex__", None)
        return '+'.join([
            ((latex(c) if latex is not None else str(c)) if c != self._ring.one or k == 0 else '') +
            (f'{self._var.__latex__()}' if k != 0 else '') +
            (f'^{{{k}}}' if k not in (0, 1) else '')
            for k, c in reversed(list(enumerate(self._coefficients)))
            if c != self._ring.zero
        ]).replace('+-', '-')

    def _repr_latex_(self):
        return f"${self.__latex__()}$"

    def __repr__(self):
        return ' + '.join([
            (str(c) if c != self._ring.one or k == 0 else '') +
            (str(self._var) if k != 0 else '') +
            (f'^{k}' if k not in (0, 1) else '')
            for k, c in reversed(list(enumerate(self._coefficients)))
            if c != self._ring.zero
        ]).replace('+ -', '- ')
