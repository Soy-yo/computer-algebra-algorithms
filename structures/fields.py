import abc

from .domains import EuclideanDomain


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

    def is_prime(self, p):
        # Fields have no primes
        return False

    def factor(self, a):
        # Unit factorization is an empty product
        return []

    def value(self, a):
        return 1 if a != self.zero else 0

    def divides(self, a, b):
        # All elements but 0 divide each other
        return a != self.zero

    # Doesn't make sense to implement these methods here
    def gcd(self, a, b):
        return NotImplemented

    def lcm(self, a, b):
        return NotImplemented

    def bezout(self, a, b):
        return NotImplemented
