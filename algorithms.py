from functools import reduce

import numpy as np

from compalg.integers import IZ, ModuloIntegers
from compalg.multinomial import Multinomial


def solve_congruences(a, b, n=None):
    """
    Solves the congruence system ax ≡ b mod n. Returns an integer x such that a_i x ≡ b_i mod n_i for all i. As a
    requirement, all n_i must be pairwise coprime and all a_i must be invertible modulo n_i.
    The arguments passed to this function can be either (a, b, n) or (b, n). In the latter case a_i is assumed to
    be 1 for all i.
    :param a: [int] - all x coefficients
    :param b: [int] - all independent terms
    :param n: [int] - all modulus
    :return: (int, ModuloIntegers) - an element x that matches all congruences and its modulo ring (that is,
                                     the solution to the system is x + kN for all integer k where N is the modulo of
                                     the returned ring)
    """
    if n is None:
        n = b
        b = a
        a = np.ones(shape=(len(b),), dtype=np.int64)

    if len(a) != len(b) or len(a) != len(n) or len(b) != len(n):
        raise ValueError(f'must give same number of as, bs and ns (got {len(a)}, {len(b)}, {len(n)})')
    if any(not isinstance(ai, (int, np.integer)) for ai in a):
        raise ValueError("all as must be integers")
    if any(not isinstance(bi, (int, np.integer)) for bi in b):
        raise ValueError("all bs must be integers")
    if any(not isinstance(ni, (int, np.integer)) for ni in n):
        raise ValueError("all ns must be integers")

    a = np.array(a, dtype=np.int64)
    b = np.array(b, dtype=np.int64)
    n = np.array(n, dtype=np.int64)

    rings = [ModuloIntegers(ni) for ni in n]

    inverses = np.array([ring.inverse(ai) if ai != 1 else 1 for (ai, ring) in zip(a, rings)], dtype=np.int64)
    b = (inverses * b) % n

    if len(b) == 1:
        return b[0], ModuloIntegers(n[0])

    lcm = reduce(IZ.lcm, n)
    ring = ModuloIntegers(lcm)

    product = n.prod()

    result = 0
    for (bi, ni) in zip(b, n):
        ri = product // ni
        gcd, (_, y) = IZ.bezout(ni, ri)
        if gcd != 1:
            raise ValueError("not possible: ns must be coprime")
        result += y * ri * bi

    return result @ ring, ring


def multidivision(f, fs, field):
    """
    Computes the multidivision between f and a set of polynomials f1, ..., fk in K[x1, ..., xr], that is,
    a set of quotients q1, ..., qk and a remainder r such that f = q1 f1 + ... + qk fk + r.
    :param f: Multinomial - dividend
    :param fs: [Multinomial] - divisors
    :param field: Field - base field where the polynomials live in (the K mentioned above)
    :return: (Multinomial, ..., Multinomial) - a tuple containing k quotients in the same order as the divisors and the
                                               remainder as its last component
    """
    h = f

    def get_next():
        hexps = h.degree_exp
        for i, g in enumerate(fs):
            gexps = g.degree_exp
            if all(ge <= he for ge, he in zip(gexps, hexps)):
                return i
        return None

    aa = [field.zero for _ in range(len(fs))]
    r = field.zero
    while h != field.zero:
        i = get_next()
        while i is not None:
            hexps = h.degree_exp
            hlc = h.leading_coefficient

            fi = fs[i]
            flc = fi.leading_coefficient
            fexps = fi.degree_exp

            # d = lt(h) / lt(fi)
            dlc = field.div(hlc, flc)
            dexps = tuple(he - fe for fe, he in zip(fexps, hexps))
            d = Multinomial({dexps: dlc}, h.variables)
            aa[i] = (aa[i] + d) @ field
            h = (h - d * fi) @ field

            i = get_next()

        hlt = h.leading_term
        r = (r + hlt) @ field
        # No need to fix anything here
        h -= hlt

    return tuple(aa + [r])


def groebner_basis(fs, field):
    """
    Returns the Gröbner basis of the polynomials in fs over the given field, that is, a set of polynomials that
    generate the leading ideal containing those polynomials ({lt(f) : f in fs}).
    :param fs: [Multinomial] - polynomials contained in the leading ideal
    :param field: Field - base field where the polynomials live in
    :return: [Multinomiañ] - the Grôbner basis of the given polynomials
    """

    def spoly(f, g):
        alpha = f.degree_exp
        beta = g.degree_exp
        gamma = tuple(max(ai, bi) for ai, bi in zip(alpha, beta))
        exp1 = tuple(gi - ai for ai, gi in zip(alpha, gamma))
        exp2 = tuple(gi - bi for bi, gi in zip(beta, gamma))
        h1 = Multinomial({exp1: 1}, f.variables)
        h2 = Multinomial({exp2: 1}, g.variables)
        return (g.leading_coefficient * h1 * f - f.leading_coefficient * h2 * g) @ field

    basis = fs[:]
    p = [(fs[i], fs[j]) for i in range(len(fs)) for j in range(i + 1, len(fs))]
    while p:
        f, g = p.pop()
        h = multidivision(spoly(f, g), basis, field)[-1]
        if h != 0:
            p.extend([(h, f) for f in basis])
            basis.append(h)

    return basis


def in_ideal(f, fs, field):
    """
    Determines if f is contained in the leading ideal that contains all polynomials in fs.
    :param f: Multinomial - polynomial to be checked
    :param fs: [Multinomial] - polynomials in the leading ideal
    :param field: Field - base field where the polynomials live in
    :return: bool - True if f is contained in the ideal, False otherwise
    """
    basis = groebner_basis(fs, field)
    return multidivision(f, basis, field)[-1] == 0
