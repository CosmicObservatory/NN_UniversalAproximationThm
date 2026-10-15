import numpy as np


def partial(func, rspct, v, tol=10**(-5)):
    d = len(v)
    assert 0 < rspct <= d, "Revisa la variable a derivar"
    e_i = np.array([1 if i== rspct else 0 for i in range(1, d+1)])
    df = func(v + e_i * tol) - func(v)
    pr = tol ** (-1) * df
    return pr


def gradient(func, v):
    d = len(v)
    # Usaremos la tolerancia definida en la derivada parcial.
    gd = np.array([partial(func, i + 1, v) for i in range(d)])
    return gd


def riemann(interval: tuple, func, N=100):
    # Integración de Riemman 
    a,b = interval
    part = np.linspace(a,b, N)
    int_v = func(part) * (b - a)
    return int_v.mean()