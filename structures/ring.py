import abc
from functools import reduce

# TODO quizá añadir más cosas del libro azul, como dominios, anillos de división y cuerpos (este último seguro)


class Ring(abc.ABC):

    def __init__(self, element_type):
        self._element_type = element_type

    @property
    def element_type(self):
        """
        Tipo de los elementos de este anillo.
        :return: type - tipo de los elementos de este anillo
        """
        return self._element_type

    def add(self, a, b):
        """
        Operación aditiva. Se debe asegurar que sea asociativa (a + (b + c) = (a + b) + c), conmutativa (a + b = b + a),
        poseer un elemento neutro 0 (método zero()) y cada elemento un inverso (\forall a \exists b : a + b = 0).
        :param a: self.element_type - elemento de la izquierda de la suma
        :param b: self.element_type - elemento de la derecha de la suma
        :return: self.element_type - la suma a + b
        """
        raise NotImplementedError

    def mul(self, a, b):
        """
        Operación multiplicativa. Se debe asegurar que sea asociativa (a * (b * c) = (a * b) * c) y distributiva para la
        suma (a * (b + c) = a * b + a * c , (a + b) * c = a * c + b * c). Si además fuera conmutativo el anillo sería un
        anillo conmutativo.
        :param a: self.element_type - elemento de la izquierda del producto
        :param b: self.element_type - elemento de la derecha del producto
        :return: self.element_type - el producto a * b
        """
        raise NotImplementedError

    @property
    def zero(self):
        """
        Elemento neutro para la suma.
        :return: self.element_type - elemento neutro para la suma 0 que cumple 0 + a = a \forall a
        """
        raise NotImplementedError

    def negate(self, a):
        """
        Devuelve el elemento opuesto para la suma para el elemento a. Esto es, el elemento b que cumple a + b = 0.
        :param a: self.element_type - elemento del que se quiere obtener el opuesto
        :return: self.element_type - el elemento opuesto de a, b = -a, que cumple a + b = 0
        """
        raise NotImplementedError

    def subs(self, a, b):
        """
        Devuelve la resta entre los elementos a y b, definida como a + (-b). Las subclases pueden sobreescribir este
        método para hacerlo más eficiente.
        :param a: self.element_type - elemento de la izquierda de la resta
        :param b: self.element_type - elemento de la derecha de la resta
        :return: self.element_type - la resta a - b
        """
        self._check_params(a, b)
        return self.add(a, self.negate(b))

    def times(self, a, n):
        """
        Devuelve el resultado de sumar n veces el elemento por sí mismo, es decir, \sum_{i=1}^n a. En algunos anillos
        coincidirá con mul(a, b).
        :param a: self.element_type - elemento a sumar
        :param n: int - número de veces a sumar (n >= 0)
        :return: self.element_type - \sum_{i=1}^n a
        """
        self._check_params(a)
        return reduce(lambda x, y: self.sum(x, y), (a for _ in range(n)), self.zero)

    # TODO merece la pena usar fast exp?
    def pow(self, a, n):
        """
        Devuelve el resultado de multiplicar n veces el elemento por sí mismo, es decir, \product_{i=1}^n a.
        :param a: self.element_type - elemento a multiplicar
        :param n: int - número de veces a multiplicar (n > 0)
        :return: self.element_type - \product_{i=1}^n a
        """
        self._check_params(a)
        return reduce(lambda x, y: self.mul(x, y), (a for _ in range(n)))

    @property
    def is_commutative(self):
        """
        Determina si el anillo es conmutativo. Por defecto devuelve siempre False y son las subclases las que deben
        sobreescribir la propiedad en caso de que sea un anillo conmutativo.
        :return: bool - si el anillo es conmutativo o no
        """
        return False

    def _check_params(self, *args):
        for a in args:
            if not isinstance(a, self._element_type):
                raise ValueError(f"expected {self._element_type} but got {type(a)}: {a}")


# TODO cosas relacionadas con divisores de 0 etc?
class UnitaryRing(Ring, abc.ABC):

    @property
    def one(self):
        """
        Elemento neutro para el producto.
        :return: self.element_type - elemento neutro para el producto 1 que cumple 1 * a = a \forall a
        """
        raise NotImplementedError

    def is_unit(self, a):
        """
        Determina si el elemento a es una unidad de este anillo, esto es, si existe un elemento inverso para a (un
        elemento b que cumpla a * b = 1).
        :param a: self.element_type - elemento a comprobar si es unidad
        :return: bool - si el elemento es unidad o no
        """
        raise NotImplementedError

    def inverse(self, a):
        """
        Devuelve el elemento inverso para el producto para el elemento a, en caso de que este exista. Esto es, el
        elemento b que cumple a * b = 1.
        :param a: self.element_type - elemento del que se quiere obtener el inverso
        :return: self.element_type - el elemento inverso de a, b = a^-1, que cumple a * b = 1
        :raise: ValueError si el elemento no es unidad
        """
        raise NotImplementedError

    def pow(self, a, n):
        """
        Devuelve la potencia a^n. A diferencia del método original (en Ring), este sí admite n <= 0. Si n = 0,
        el resultado es self.one y si n < 0 el resultado es (a^-1)^n, siempre que a sea unidad.
        :param a: self.element_type - elemento a multiplicar
        :param n: int - número de veces a multiplicar
        :return: self.element_type - \product_{i=1}^n a
        """
        self._check_params(a)
        if n < 0:
            a = self.inverse(a)
            n = -n
        return reduce(lambda x, y: self.mul(x, y), (a for _ in range(n)), self.one)
