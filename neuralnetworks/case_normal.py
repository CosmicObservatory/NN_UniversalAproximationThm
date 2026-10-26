import numpy as np
from neuralnetworks.maths import riemann
# Función a estimar. 
def densnormal(x):
    # Función de densidad de N(0,1)
    c = 1/np.sqrt(2*np.pi)
    y = c * np.exp(-0.5 * x ** 2)
    return y

def fdist(x):
    # Función de distribución. Fijada para una normal
    return riemann((-10, x), func=densnormal)


def error_learn(ty, tx, model, theta):
    err = MSE_nn(ty, tx, m=model, pars=theta)
    return err

