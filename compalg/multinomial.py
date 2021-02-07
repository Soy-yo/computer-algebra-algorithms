import numpy as np

from compalg.polynomials import Var


def link(*variables):
    """
    Links the given variables so they can be used together to create a Multinomial. It is not possible to create
    multinomials with non-linked variables. As Multinomial does not rely on the Var class it is not necessary to pass
    actual Var instance as parameters of this function, but they are admissible. For example:
    >>> x = Var('x')
    >>> x, y = link(x, 'y')  # Now x and y are Multinomial instances that can (only) use variables 'x' and 'y'
    >>> x ** 2 + x * y + 1  # x^2 + xy + 1
    >>> z = Var('z')
    >>> x + z  # Error: z has not been linked with x!
    :param variables: Var or object - variables (or their representations) to be linked together
    :return: [Multinomial] - Multinomial representation of each variable in the same order as they were passed as
                             parameters
    """
    n = len(variables)
    # Sort vars and remember their original order
    poly_vars = tuple(sorted((v.x, i) if isinstance(v, Var) else (v, i) for i, v in enumerate(variables)))
    # Use only variables to create polynomials
    variables = tuple(x for x, i in poly_vars)
    coeffs = [{(0,) * i + (1,) + (0,) * (n - i - 1): 1} for i in range(n)]
    polys = [Multinomial(cs, variables) for cs in coeffs]
    # Return the polynomials in the original order
    result = [None] * n
    for (p, (_, i)) in zip(polys, poly_vars):
        result[i] = p
    return result


_ordering = 'dp'


def set_ordering(o):
    """
    Sets the ordering used to compute leading terms in polynomials.
    :param o: str - new ordering: one of dp (degree ordering, initial) or lp (lexicographic order)
    """
    if o not in ('dp', 'lp'):
        raise ValueError(f"ordering must be either dp or lp (received {o})")
    global _ordering
    _ordering = o


class Multinomial:
    """
    Class representing a multivariate ploynomial sum(x^alpha; alpha) for x = (x_1, ..., x_n) and
    alpha = (alpha_1, ..., alpha_n).
    """

    def __init__(self, coefficients, variables):
        """
        :param coefficients: {(int, ..., int) -> any} - coefficients of the polynomial, where keys represent the
                                                        exponent of each var in the same order
        :param variables: (any, ..., any) - variable representations sorted
        """
        # Make sure there are no zero coefficients
        self._coefficients = {e: c for e, c in coefficients.items() if c != 0}
        self._variables = variables
        # Lazy properties
        self._degree = None
        self._lt = None

    @property
    def variables(self):
        """
        Returns the variables of this polynomial.
        :return: (any, ..., any) - tuple of variables
        """
        return self._variables

    @property
    def degree(self):
        """
        Returns the degree of this polynomial, that is, the maximum exponent of any term.
        :return: int - degree of this polynomial
        """
        if self._degree is None:
            if len(self._coefficients) == 0:
                self._degree = -1
            else:
                self._degree = max(sum(exps) for exps in self._coefficients.keys())

        return self._degree

    @property
    def degree_exp(self):
        """
        Returns the exponent associated with the term with highest degree.
        :return: (int, ..., int) - exponents of the term with highest degree
        """
        lt = self.leading_term
        if lt == 0:
            return (0,) * len(self._variables)
        exp, = lt._coefficients.keys()
        return exp

    @property
    def leading_term(self):
        """
        Returns the leading term of this polynomial, that is the term with highest degree.
        :return: Multinomial - leading term as polynomial
        """
        if self._lt is None:
            if not self._coefficients:
                self._lt = Multinomial({}, self._variables)
                return self._lt
            # Get the maximum term relatively to the ordering
            key = (lambda x: (sum(x[0]), x[0])) if _ordering == 'dp' else None
            e, c = max(self._coefficients.items(), key=key)
            self._lt = Multinomial({e: c}, self._variables)

        return self._lt

    @property
    def leading_coefficient(self):
        """
        Returns the leading coefficient of this polynomial, that is, the coefficient of the leading term.
        :return: int - leading coefficient
        """
        ((_, c),) = self.leading_term._coefficients.items()
        return c

    @property
    def leading_monomial(self):
        """
        Returns the leading monomial of this polynomial, that is, the leading term but with 1 as coefficient.
        :return: Multinomial - leading monomial
        """
        ((e, _),) = self.leading_term._coefficients.items()
        return Multinomial({e: 1}, self._variables)

    @property
    def independent_term(self):
        """
        Returns the independent term of this polynomial, that is, the term with (0, ..., 0) as exponent.
        :return: int - independent term
        """
        key = (0,) * len(self._variables)
        if key in self._coefficients:
            return self._coefficients[key]
        return 0

    @property
    def terms(self):
        """
        Returns the number of terms in this polynomial.
        :return: int - number of terms in this polynomial
        """
        return len(self._coefficients)

    def __len__(self):
        return self.terms

    def used_vars(self):
        """
        Returns a tuple with the vars that are effectively being used in this polynomial. For instance,
        (f(x, y, z) = x^2 + z).used_vars() == (x, z).
        :return: (any, ..., any) - all vars that are being used by this polynomial
        """
        result = [False] * len(self._variables)
        for e in self._coefficients.keys():
            for i, ei in enumerate(e):
                if ei > 0:
                    result[i] = True
        return tuple(self._variables[b] for b in result)

    def without_vars(self, *xs):
        """
        Removes the given variables from this polynomial. This method will raise an error if the variable is present
        inside any term of this polynomial
        :param xs: int, Var, Multinomial, any - indices or variables to remove
        :return: Multinomial - the same polynomial expressed without the given vars
        """
        xs = [self._get_var(x) for x in xs]
        ii = {self._variables.index(x) for x in xs}
        coeffs = dict()
        for e, c in self._coefficients.items():
            for i, x in zip(ii, xs):
                if e[i] != 0:
                    raise ValueError(f"cannot remove variable {x} as it is being used")
            # Remove not necessary exponents
            e_ = tuple(ei for i, ei in enumerate(e) if i not in ii)
            coeffs[e_] = c

        variables = tuple(x for i, x in enumerate(self._variables) if i not in ii)
        return Multinomial(coeffs, variables)

    def with_vars(self, *xs):
        """
        Adds the given variables to this polynomial. These variables cannot be already present in this polynomial.
        :param xs: Var, Multinomial, any - variables to be added to this polynomial
        :return: Multinomial - the same polynomial expressed with the given vars
        """
        xs = tuple(self._get_var(x) for x in xs)
        variables = tuple(sorted(self._variables + xs))
        n = len(variables)
        # Indices of old variables
        ii = [variables.index(x) for x in self._variables]
        coeffs = dict()
        for e, c in self._coefficients.items():
            e_ = [0] * n
            for j, i in enumerate(ii):
                e_[i] = e[j]
            coeffs[tuple(e_)] = c

        return Multinomial(coeffs, variables)

    def __add__(self, q):
        coeffs = self._coefficients.copy()
        if not isinstance(q, Multinomial):
            it = self.independent_term
            exps = (0,) * len(self._variables)
            coeffs[exps] = it + q
            return Multinomial(coeffs, self._variables)

        if self._variables != q._variables:
            raise ValueError("can only operate polynomials with the same variables")

        for e, c in q._coefficients.items():
            if e in coeffs:
                coeffs[e] += c
            else:
                coeffs[e] = c
        return Multinomial(coeffs, self._variables)

    def __radd__(self, q):
        coeffs = self._coefficients.copy()
        it = self.independent_term
        exps = (0,) * len(self._variables)
        coeffs[exps] = q + it
        return Multinomial(coeffs, self._variables)

    def __neg__(self):
        return Multinomial({e: -c for e, c in self._coefficients.items()}, self._variables)

    def __sub__(self, q):
        return self + (-q)

    def __rsub__(self, q):
        return q + (-self)

    def __mul__(self, q):
        def exp(alpha, beta):
            return tuple(a + b for a, b in zip(alpha, beta))

        if not isinstance(q, Multinomial):
            coeffs = {e: c * q for e, c in self._coefficients.items()}
            return Multinomial(coeffs, self._variables)

        if self._variables != q._variables:
            raise ValueError("can only operate polynomials with the same variables")

        coeffs = dict()
        for e1, c1 in self._coefficients.items():
            for e2, c2 in q._coefficients.items():
                c = c1 * c2
                if c != 0:
                    e = exp(e1, e2)
                    if e in coeffs:
                        # Might be 0 now but it will be removed inside __init__
                        coeffs[e] += c
                    else:
                        coeffs[e] = c

        return Multinomial(coeffs, self._variables)

    def __rmul__(self, q):
        coeffs = {e: q * c for e, c in self._coefficients.items()}
        return Multinomial(coeffs, self._variables)

    def __pow__(self, n):
        def pow_(p, k):
            if k == 1:
                return Multinomial(self._coefficients, self._variables)
            if k % 2 == 0:
                return pow_(p, k // 2) * pow_(p, k // 2)
            return p * pow_(p, k - 1)

        if isinstance(n, (float, np.float)) and not n.is_integer():
            raise ValueError("exponent must be an integer")
        if not isinstance(n, (int, np.integer)):
            raise ValueError("exponent must be an integer")

        if n < 0:
            raise ValueError("exponent must be positive or zero")
        if n == 0:
            return Multinomial({(0,) * len(self._variables): 1}, self._variables)

        return pow_(self, n)

    def __eq__(self, q):
        if not isinstance(q, Multinomial):
            if self.degree > 0:
                return False
            return self.independent_term == q

        return self._variables == q._variables and self._coefficients == q._coefficients

    def __hash__(self):
        return hash((self._variables, tuple(self._coefficients.items())))

    def __call__(self, *args, **kwargs):
        """
        Substitutes the given variables in this polynomial and return the result. First args are treated as
        substitutions in the same order as the variables appear in the variables' property. Keyword args are treated
        as substitutions of the variable with the given name. For instance, if p = x^2 + y, p(2, 5) = 9, p(2) = y + 4,
        p(y=5) = x + 5 and p(2, x=3) = ERROR!!
        :param args: number - value to be replaced in the given variable (substitution of variables will not work)
        :return: Multinomial - polynomial after the substitutions have been done
        """
        values = {i: v for i, v in enumerate(args)}
        for x, v in kwargs.items():
            i = self._variables.index(x)
            if i in values:
                raise ValueError(f"multiple values given for variable {x}")
            values[i] = v

        coeffs = dict()
        for e, c in self._coefficients.items():
            w = c
            e_ = list(e)
            for i, v in values.items():
                k = e_[i]
                w *= v ** k
                e_[i] = 0

            e_ = tuple(e_)
            if e_ not in coeffs:
                coeffs[e_] = 0
            coeffs[tuple(e_)] += w

        return Multinomial(coeffs, self._variables)

    def __matmul__(self, ring):
        # Calls @ for all coefficients
        return Multinomial({e: c @ ring for e, c in self._coefficients.items()}, self._variables)

    def derivative(self, x):
        """
        Returns the derivative of this polynomial over the given variable.
        :param x: int, Var, Multinomial, any - if int, the index of the variable over which calculate the derivative;
                                               if Multinomial, it can only contain one term of degree and coefficient 1
                                               and will be treated as a variable
                                               if Var or any, the variable represented by it
        :return: Multinomial - the derivative of this polynomial over the given var
        """
        x = self._get_var(x)
        i = x if isinstance(x, (int, np.integer)) else self._variables.index(x)
        if i < 0:
            raise ValueError(f'{x} is not a valid variable for derivation')
        coeffs = dict()
        for alpha, c in self._coefficients.items():
            k = alpha[i]
            if k > 0:
                # If k == 0 then this term is a constant for x
                beta = alpha[:i] + (k - 1,) + alpha[i + 1:]
                coeffs[beta] = c * k
        return Multinomial(coeffs, self._variables)

    def gradient(self):
        """
        Returns the gradient of this polynomial, that is, the vector of the n possible derivatives where n is the
        number of variables this polynomial has, sorted by the same order the variables have. For example, for
        x^2z + 2z in IZ[x, y, z], its gradient is (2xz, 0, x^2 + 2).
        :return: (Multinomial, ..., Multinomial) - a tuple containing all the derivatives with the specified order
        """
        return tuple(self.derivative(i) for i in range(len(self._variables)))

    @staticmethod
    def _get_var(x):
        if isinstance(x, Var):
            # If Var get its representation
            return x.x
        if isinstance(x, Multinomial) and len(x) == 1:
            # If we were given a Multinomial as variable check it is actually a variable
            ((alpha, c),) = x._coefficients.items()
            if c == 1 and sum(alpha) == 1:
                return x._variables[alpha.index(1)]
            else:
                raise ValueError(f"invalid Multinomial as variable: {x}")

        return x

    def _repr_coefficient(self, c, alpha, latex):
        isit = all(a == 0 for a in alpha)
        if c == 1 and not isit:
            return ''
        if c == -1 and not isit:
            return '-'
        r = getattr(c, '__latex__', lambda: repr(c)) if latex else lambda: repr(c)
        if hasattr(c, '__len__') and len(c) > 1:
            return '(' + r() + ')'
        return r()

    def _repr_vars(self, alpha, latex):
        def repr_var(x, k):
            if k == 0:
                return ''
            if k == 1:
                return f'{x}'
            if latex:
                return f'{x}^{{{k}}}'
            return f'{x}^{k}'

        if all(a == 0 for a in alpha):
            return ''

        return ''.join([repr_var(x, a) for a, x in zip(alpha, self._variables)])

    def __latex__(self):
        return self.__repr__(latex=True)

    def _repr_latex_(self):
        return f"${self.__latex__()}$"

    def __repr__(self, latex=False):
        if self.degree == -1:
            return '0'

        coeffs = sorted(self._coefficients.items(), key=lambda x: (sum(x[0]), x), reverse=True)

        return ' + '.join([
            self._repr_coefficient(c, alpha, latex=latex) + self._repr_vars(alpha, latex=latex)
            for alpha, c in coeffs
        ]).replace('+ -', '- ')
