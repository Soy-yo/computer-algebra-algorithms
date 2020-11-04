import abc

from .rings import DivisionRing, CommutativeRing
from .domains import Domain


class Field(DivisionRing, CommutativeRing, Domain, abc.ABC):
    """
    Class representing a field, a commutative ring for which all elements are units but 0.
    """
