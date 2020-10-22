import abc
from functools import reduce

# TODO quizá añadir más cosas del libro azul, como dominios, anillos de división y cuerpos (este último seguro)


class Ring(abc.ABC):
    """
    Class representing a ring. It consists of a set equipped with two binary operations that generalize the arithmetic
    operations of addition and multiplication. This class should not be instantiated directly, rather it should be
    extended with subclasses.
    """

    def __init__(self, element_type):
        """
        :param element_type: type - type of the elements contained in this ring
        """
        self._element_type = element_type

    @property
    def element_type(self):
        """
        Type of the elements contained in this ring.
        :return: type - type of the elements contained in this ring
        """
        return self._element_type

    def add(self, a, b):
        """
        Additive operation. It must be associative (a + (b + c) = (a + b) + c), commutative (a + b = b + a),
        there must be an identity element 0 (property self.zero) such that a + 0 = a for all a and every element must
        have an opposite (for all a there exists an additive inverse element b such that a + b = 0).
        :param a: self.element_type - left-hand-side element
        :param b: self.element_type - right-hand-side element
        :return: self.element_type - the sum a + b
        """
        raise NotImplementedError

    def mul(self, a, b):
        """
        Multiplicative operation. It must be associative (a * (b * c) = (a * b) * c) and distributive wrt the sum
        (a * (b + c) = a * b + a * c , (a + b) * c = a * c + b * c). If this operation is also commutative the ring
        is called a commutative ring.
        :param a: self.element_type - left-hand-side element
        :param b: self.element_type - right-hand-side element
        :return: self.element_type - the product a * b
        """
        raise NotImplementedError

    @property
    def zero(self):
        """
        Identity element for the sum.
        :return: self.element_type - identity element for the sum 0 such that 0 + a = a for all a
        """
        raise NotImplementedError

    def negate(self, a):
        """
        Returns the opposite element for the sum of the given element a. That is, en element b such that a + b = 0.
        :param a: self.element_type - element for which return the opposite
        :return: self.element_type - a's opposite element: b = -a, such that a + b = 0
        """
        raise NotImplementedError

    def sub(self, a, b):
        """
        Returns the result of subtracting b to a. The subtraction operation is defined as a + (-b), where -b is the
        opposite of b. Subclasses may override this method to make it more efficient.
        :param a: self.element_type - left-hand-side element
        :param b: self.element_type - right-hand-side element
        :return: self.element_type - the subtraction a - b
        """
        self._check_params(a, b)
        return self.add(a, self.negate(b))

    def times(self, a, n):
        """
        Returns the result of adding n times the element a to itself, that is, the sum from 1 to n of a.
        :param a: self.element_type - element to be added
        :param n: int - number of times to add the element (n >= 0)
        :return: self.element_type - sum(1, n; a) = na
        """
        self._check_params(a)
        return reduce(lambda x, y: self.sum(x, y), (a for _ in range(n)), self.zero)

    # TODO merece la pena usar fast exp?
    def pow(self, a, n):
        """
        Returns the result of multiplying n times a to itself, that is, the product from 1 to n of a.
        :param a: self.element_type - element to be multiplied
        :param n: int - number of times to multiply the element (n > 0)
        :return: self.element_type - product(1, n; a) = a^n
        """
        self._check_params(a)
        return reduce(lambda x, y: self.mul(x, y), (a for _ in range(n)))

    @property
    def is_commutative(self):
        """
        Property that determines if this ring is commutative or not. By default it is always False and it is up to
        subclasses to override it if necessary.
        :return: bool - True if this ring is commutative, False otherwise
        """
        return False

    def _check_params(self, *args):
        for a in args:
            if not isinstance(a, self._element_type):
                raise ValueError(f"expected {self._element_type} but got {type(a)}: {a}")


# TODO cosas relacionadas con divisores de 0 etc?
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
        :return: self.element_type - identity element for the product 1 such that 1 * a = a for all a
        """
        raise NotImplementedError

    def is_unit(self, a):
        """
        Determines whether a is an unit of this ring or not, that is, if there exits an inverse element of a for the
        product (some b such that a * b = 1).
        :param a: self.element_type - element to be checked
        :return: bool - True if a is an unit, False otherwise
        """
        raise NotImplementedError

    def inverse(self, a):
        """
        Returns the inverse element of a for the product, assuming there exits one. That is, the element b such that
        a * b = 1.
        :param a: self.element_type - element for which calculate its inverse
        :return: self.element_type - the inverse element of a: b = a^-1, such that a * b = 1
        :raise: ValueError - if a is not an unit
        """
        raise NotImplementedError

    def pow(self, a, n):
        """
        Returns the power a^n. It differs from the original method (in Ring) in the way that this method does allow
        n <= 0. If n = 0 self.one will be returned and if n < 0 the result will be (a^-1)^n as long as a is an unit.
        :param a: self.element_type - element to be multiplied
        :param n: int - number of times to multiply the element
        :return: self.element_type - product(1, n; a) = a^n
        :raise: ValueError - if n < 0 and a is not an unit
        """
        self._check_params(a)
        if n < 0:
            a = self.inverse(a)
            n = -n
        return reduce(lambda x, y: self.mul(x, y), (a for _ in range(n)), self.one)
