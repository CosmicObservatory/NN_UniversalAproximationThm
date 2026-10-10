"""
    Ejemplo de Red Neuronal.
    1 Neurona Salida
    1 Neurona Oculta
    1 Neurona Salida
    
    Problema ejemplo.
    Aproximar la funcion indicadora
"""

# Basic Functions

import numpy as np


def sigmoide(x):
    return (1 + np.exp(-x)) ** (-1)


def model(x, pars):
    c = pars[0]
    w = pars[1]
    b = pars[2]
    v = w * x + b
    y = c * sigmoide(v)
    return y


def partial_nn(pars, var, h = 10 ** (-5), f=model):
    assert var in (0,1,2)
    vh = np.zeros(3)
    vh[var] += h
    vh += pars
    partial = h ** (-1) * (f(vh) - f(pars))
    return partial


def gradient(pars):
    grad = [partial_nn(pars, v, f=sce_nn) for v in range(3)]
    return np.array(grad)


def grad_descent(x_0, lr=0.1, t_epocs=1_000):
    for epocs in range(t_epocs):
        g_0 = gradient(x_0)
        x_n = x_0 - lr * g_0
        x_0 = x_n
        if epocs % 100 == 0:
            print(epocs, sce_nn(x_0))
    print("Parametro final",x_0, "Error Final", sce_nn(x_0))
    return x_0

def sce_nn(pars,N=2, Np=20):
    x = np.linspace(-N, N, Np)
    y = np.int16(x > 0)
    error = (y - model(x, pars)) ** 2
    return Np ** (-1) * error.sum()


def goal(x):
    return 1 if x>0 else 0
