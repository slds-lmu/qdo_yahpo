import numpy as np

def trafo(x, lower, upper):
    n = len(x)
    assert(n == len(lower))
    assert(n == len(upper))
    return [(x[i] - lower[i]) / (upper[i] - lower[i]) for i in range(n)]

def retrafo(z, lower, upper):
    n = len(z)
    assert(n == len(lower))
    assert(n == len(upper))
    return [np.max((lower[i], np.min(((z[i] * (upper[i] - lower[i])) + lower[i], upper[i])))) for i in range(n)]

