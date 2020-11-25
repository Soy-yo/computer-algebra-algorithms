import abc

from .domains import Domain
from .polynomials import PolynomialField
from .rings import DivisionRing, CommutativeRing


class Field(DivisionRing, CommutativeRing, Domain, abc.ABC):
    """
    Class representing a field, a commutative ring for which all elements are units but 0.
    """

    def __getitem__(self, var):
        return PolynomialField(self, var)
