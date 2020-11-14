from functools import reduce

import numpy as np

from structures.integers import IZ, ModuloIntegers


def solve_congruences(a, b, n=None):
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
        return b[0]

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
