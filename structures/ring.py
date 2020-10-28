import abc
from functools import reduce


# TODO IMPORTANTE en general se supone que los anillos tienen unidad, los raros son los que no la tienen
#       podríamos unir UnitaryRing a Ring y simplificar un poco, dejarlo tal cual o renombrar Ring a Rng y el otro Ring
# https://en.wikipedia.org/wiki/Rng_(algebra)
# TODO quizá añadir más cosas del libro azul, como dominios, anillos de división y cuerpos (este último seguro)


class Ring(abc.ABC):
    """
    Class representing a ring. It consists of a set equipped with two binary operations that generalize the arithmetic
    operations of addition and multiplication. This class should not be instantiated directly, rather it should be
    extended with subclasses.
    """

    def __init__(self, dtype):
        """
        :param dtype: type - type of the elements contained in this ring
        """
        self._dtype = dtype

    @property
    def dtype(self):
        """
        Type of the elements contained in this ring.
        :return: type - type of the elements contained in this ring
        """
        return self._dtype

    def add(self, a, b):
        """
        Additive operation. It must be associative (a + (b + c) = (a + b) + c), commutative (a + b = b + a),
        there must be an identity element 0 (property self.zero) such that a + 0 = a for all a and every element must
        have an opposite (for all a there exists an additive inverse element b such that a + b = 0).
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side element
        :return: self.dtype - the sum a + b
        """
        raise NotImplementedError

    def mul(self, a, b):
        """
        Multiplicative operation. It must be associative (a * (b * c) = (a * b) * c) and distributive wrt the sum
        (a * (b + c) = a * b + a * c , (a + b) * c = a * c + b * c). If this operation is also commutative the ring
        is called a commutative ring.
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side element
        :return: self.dtype - the product a * b
        """
        raise NotImplementedError

    @property
    def zero(self):
        """
        Identity element for the sum.
        :return: self.dtype - identity element for the sum 0 such that 0 + a = a for all a
        """
        raise NotImplementedError

    def negate(self, a):
        """
        Returns the opposite element for the sum of the given element a. That is, en element b such that a + b = 0.
        :param a: self.dtype - element for which return the opposite
        :return: self.dtype - a's opposite element: b = -a, such that a + b = 0
        """
        raise NotImplementedError

    def sub(self, a, b):
        """
        Returns the result of subtracting b to a. The subtraction operation is defined as a + (-b), where -b is the
        opposite of b. Subclasses may override this method to make it more efficient.
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side element
        :return: self.dtype - the subtraction a - b
        """
        self._check_params(a, b)
        return self.add(a, self.negate(b))

    def times(self, a, n):
        """
        Returns the result of adding n times the element a to itself, that is, the sum from 1 to n of a.
        :param a: self.dtype - element to be added
        :param n: int - number of times to add the element (n >= 0)
        :return: self.dtype - sum(1, n; a) = na
        """
        self._check_params(a)
        return reduce(lambda x, y: self.sum(x, y), (a for _ in range(n)), self.zero)

    def pow(self, a, n):
        """
        Returns the result of multiplying n times a to itself, that is, the product from 1 to n of a.
        :param a: self.dtype - element to be multiplied
        :param n: int - number of times to multiply the element (n > 0)
        :return: self.dtype - product(1, n; a) = a^n
        """
        self._check_params(a)
        return reduce(lambda x, y: self.mul(x, y), (a for _ in range(n)))

    def is_idempotent(self, a):
        """
        Determines if the given element is idempotent, that is, a^2 = a.
        :param a: self.dtype - element to be checked
        :return: bool - whether a is idempotent or not
        """
        return self.mul(a, a) == a

    @property
    def char(self):
        """
        Property that tells the characteristic of the ring, that is, the minimum integer n such that na = 0 for some
        element a. If such n does not exist, char = 0.
        :return: int - characteristic of this ring
        """
        raise NotImplementedError

    def __latex__(self):
        """
        Returns the LaTeX representation of this class. This method is here only so subclasses don't forget to add the
        LaTeX representation.
        :return: str - LaTeX representation of this class
        """
        # If there's no such LaTeX representation just return None (or leave it unimplemented)
        raise NotImplementedError

    def _repr_latex_(self):
        # Do not override this method
        return self.__latex__()

    def _check_params(self, *args):
        for a in args:
            # FIXME it may not work well with numpy (np.int32/64, np.uint32/64 or np.integer are not the same as int)
            if not isinstance(a, self._dtype):
                raise ValueError(f"expected {self._dtype} but got {type(a)}: {a}")


class UnitaryRing(Ring, abc.ABC):
    """
    Class representing a special type of ring: an unitary ring. An unitary ring is a ring which also contains a
    multiplicative identity element. With this, elements may have a multiplicative inverse and those elements are
    called units.
    """

    @property
    def one(self):
        """
        Identity element for the product.
        :return: self.dtype - identity element for the product 1 such that 1 * a = a for all a
        """
        raise NotImplementedError

    def is_unit(self, a):
        """
        Determines whether a is an unit of this ring or not, that is, if there exits an inverse element of a for the
        product (some b such that a * b = 1).
        :param a: self.dtype - element to be checked
        :return: bool - True if a is an unit, False otherwise
        """
        raise NotImplementedError

    def inverse(self, a):
        """
        Returns the inverse element of a for the product, assuming there exits one. That is, the element b such that
        a * b = 1.
        :param a: self.dtype - element for which calculate its inverse
        :return: self.dtype - the inverse element of a: b = a^-1, such that a * b = 1
        :raise: ValueError - if a is not an unit
        """
        raise NotImplementedError

    def pow(self, a, n):
        """
        Returns the power a^n. It differs from the original method (in Ring) in the way that this method does allow
        n <= 0. If n = 0 self.one will be returned and if n < 0 the result will be (a^-1)^n as long as a is an unit.
        :param a: self.dtype - element to be multiplied
        :param n: int - number of times to multiply the element
        :return: self.dtype - product(1, n; a) = a^n
        :raise: ValueError - if n < 0 and a is not an unit
        """
        self._check_params(a)
        if n < 0:
            a = self.inverse(a)
            n = -n
        return reduce(lambda x, y: self.mul(x, y), (a for _ in range(n)), self.one)

    @property
    def is_commutative(self):
        """
        Property that determines if this ring is commutative or not. By default it is always False and it is up to
        subclasses to override it if necessary.
        :return: bool - True if this ring is commutative, False otherwise
        """
        return False

    @property
    def is_domain(self):
        """
        Property that determines if this ring is a domain or not. A domain is a ring that does not contain zero
        divisors, which means that if a * b = 0, either a = 0 or b = 0. By default it is always False and it is up to
        subclasses to override it if necessary.
        :return: bool - True if this ring is a domain, False otherwise
        """
        return False


class CommutativeRing(UnitaryRing, abc.ABC):
    """
    Class representing of a commutative ring.
    """

    @property
    def is_commutative(self):
        return True


class Domain(CommutativeRing, abc.ABC):
    """
    Class representing of an integral domain.
    """

    @property
    def is_domain(self):
        return True


class UFD(Domain, abc.ABC):
    """
    Class representing of an Unique Factorization Domain, a domain where all elements can be written as a product of
    prime elements (or irreducible elements), uniquely up to order and units.
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
        # TODO break RSA :)
        raise NotImplementedError


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
        return self.full_div(a, b)[0]

    def rem(self, a, b):
        """
        Returns the remainder r of the division a = b * q + r.
        :param a: self.dtype - dividend
        :param b: self.dtype - divisor
        :return: self.dtype - remainder of the division
        """
        return self.full_div(a, b)[1]

    def full_div(self, a, b):
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

    def gcd(self, a, b):
        """
        Returns the greatest common divisor of a and b, that is, the greatest element which divides both a and b.
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side element
        :return: self.dtype - greatest common divisor of a and b
        """
        raise NotImplementedError

    def lcm(self, a, b):
        """
        Returns the least common multiple of a and b, that is, the lowest element which is divided by both a and b.
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side element
        :return: self.dtype - least common multiple of a and b
        """
        # TODO return self.quot(self.mul(a, b), self.gcd(a, b)) ?
        raise NotImplementedError

    def bezout(self, a, b):
        """
        Returns two elements that satisfy Bezout's identity, that is, two elements x and y for which
        gcd(a, b) = a * x + b * y.
        :param a: self.dtype - left-hand-side element
        :param b: self.dtype - right-hand-side element
        :return: (self.dtype, self.dtype) - two elements that satisfy Bezout's identity
        """
        raise NotImplementedError


class Field(EuclideanDomain, abc.ABC):
    """
    Class representing a field, a ring for which all elements are units but 0.
    """

    def div(self, a, b):
        """
        Alias of EuclideanDomain.quot, which makes more sense in the context of a field.
        :param a: self.dtype - dividend
        :param b: self.dtype - divisor
        :return: self.dtype - quotient of the division
        """
        if b == self.zero:
            raise ZeroDivisionError
        return self.mul(a, self.inverse(b))

    def quot(self, a, b):
        return self.div(a, b)

    def rem(self, a, b):
        return self.mod(a, b)

    def full_div(self, a, b):
        return self.div(a, b), 0

    def is_unit(self, a):
        # All elements but 0 are units
        return a != self.zero
