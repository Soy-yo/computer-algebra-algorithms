import abc
from functools import reduce


# TODO: permitir los corchetes para devolver su anillo de polinomios :O
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

    @property
    def char(self):
        """
        Property that tells the characteristic of the ring, that is, the minimum integer n such that na = 0 for some
        element a. If such n does not exist, char = 0.
        :return: int - characteristic of this ring
        """
        raise NotImplementedError

    @property
    def zero(self):
        """
        Identity element for the sum.
        :return: self.dtype - identity element for the sum 0 such that 0 + a = a for all a
        """
        raise NotImplementedError

    @property
    def is_domain(self):
        """
        Property that determines if this ring is a domain or not. A domain is a ring that does not contain zero
        divisors, which means that if a * b = 0, either a = 0 or b = 0. By default it is always False and it is up to
        subclasses to override it if necessary.
        :return: bool - True if this ring is a domain, False otherwise
        """
        return False

    @property
    def is_unitary(self):
        """
        Property determining if the ring is unitary or not.
        :return: bool - True if the ring is unitary, False otherwise
        """
        return False

    @property
    def is_commutative(self):
        """
        Property that determines if this ring is commutative or not. By default it is always False and it is up to
        subclasses to override it if necessary.
        :return: bool - True if this ring is commutative, False otherwise
        """
        return False

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

    def is_zero_divisor(self, a):
        """
        a is a zero divisor if there exists some b that verify ab = 0 or ba = 0.
        :return: bool - if it is or not
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
        a = a @ self
        b = b @ self
        return self.add(a, self.negate(b))

    def times(self, a, n):
        """
        Returns the result of adding n times the element a to itself, that is, the sum from 1 to n of a.
        :param a: self.dtype - element to be added
        :param n: int - number of times to add the element (n >= 0)
        :return: self.dtype - sum(1, n; a) = na
        """
        a = a @ self
        return reduce(lambda x, y: self.add(x, y), (a for _ in range(n)), self.zero)

    def pow(self, a, n):
        """
        Returns the result of multiplying n times a to itself, that is, the product from 1 to n of a.
        :param a: self.dtype - element to be multiplied
        :param n: int - number of times to multiply the element (n > 0)
        :return: self.dtype - product(1, n; a) = a^n
        """
        a = a @ self
        return reduce(lambda x, y: self.mul(x, y), (a for _ in range(n)))

    def is_idempotent(self, a):
        """
        Determines if the given element is idempotent, that is, a^2 = a.
        :param a: self.dtype - element to be checked
        :return: bool - whether a is idempotent or not
        """
        return self.mul(a, a) == a

    def contains(self, a):
        """
        Returns whether this ring contains the element a or not. It can be used in its operator form: a in ring.
        :param a: self.dtype - element to be checked
        :return: bool - True if this ring contains a (or another representation of a), False otherwise
        """
        raise NotImplementedError

    def at(self, a):
        """
        Converts the given element to the usual representation of that element in this ring. It may also change its
        type. It can be used in its operator form: a @ Ring.
        For example, if ring = IZ/4IZ, ring.at(13) (or 13 @ ring) will return 1.
        :param a: object - element to be converted
        :return: self.dtype - usual representation of a
        :raise: ValueError - if a cannot be converted to an element of this ring (i.e. a not in ring)
        """
        raise NotImplementedError

    def __latex__(self):
        """
        Returns the LaTeX representation of this class. This method is here only so subclasses don't forget to add the
        LaTeX representation.
        :return: str - LaTeX representation of this class
        """
        # If there's no such LaTeX representation just return NotImplemented
        raise NotImplementedError

    def _repr_latex_(self):
        # Do not override this method
        latex = self.__latex__()
        if latex == NotImplemented:
            return NotImplemented
        return f"${self.__latex__()}$"

    def __contains__(self, a):
        return self.contains(a)

    def __rmatmul__(self, a):
        return self.at(a)


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

    @property
    def is_unitary(self):
        return True

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
        a = a @ self
        if n < 0:
            a = self.inverse(a)
            n = -n
        return reduce(lambda x, y: self.mul(x, y), (a for _ in range(n)), self.one)


class CommutativeRing(Ring, abc.ABC):
    """
    Class representing a commutative ring. That is, when ab = ba for all a, b.
    """

    @property
    def is_commutative(self):
        return True


class DivisionRing(UnitaryRing, abc.ABC):
    """
    Class representing a division ring. That is, when all non-null elements are units.
    """

    @property
    def is_division_ring(self):
        return True

    def div(self, a, b):
        """
        Returns the division a/b = a * b^-1.
        :param a: self.dtype - numerator
        :param b: self.dtype - denominator
        :return: self.dtype - division
        """
        if b == self.zero:
            raise ZeroDivisionError
        return self.mul(a, self.inverse(b))

    def is_unit(self, a):
        # All elements but 0 are units
        return a != self.zero
