from neuralnetworks.maths import gradient
import numpy as np
import matplotlib.pyplot as plt


# Definimos la función de activación. El teorema nos dice que debe ser continua y de tipo sigmoidal.
def activation(x): 
    # Función Sigmoide
    y = 1 / (1 + np.exp(-x))
    return y


# Construimos la red neuronal. 
def model(x, theta, func=activation):
    # Se ve sencilla porque es una neurona.
    c = theta[0]
    w = theta[1]
    b = theta[2]
    v = w * x + b
    y = c * func(v)
    return y


# Función de Error
def MSE_nn(yv, xv, m, pars):
    M = len(yv)
    assert M == len(xv), "Revisa dimensiones"
    pred = np.array([m(x, pars) for x in xv])
    ell = (yv-pred) ** 2
    Lambda = M ** (-1) * ell.sum()
    return Lambda


# Algoritmo de Optimización
def descent_gradent(func, theta0, epsilon, stop = 100, lr = 0.01):
    error = func(theta0)
    epochs = 0
    while error > epsilon:
        if epochs > stop:
            break
        thetan = theta0 - lr * gradient(func, theta0)
        error = func(thetan)
        theta0 = thetan
        if epochs > 1000:
            print("Error:", error)
        epochs+=1
    return theta0


def verif_model(y, x, model, pars, crit):
    predicted = [model(t) for t in x]
    fig, ax = plt.subplots()
    ax.plot(x, predicted, color="blue", label="Red Neuronal")
    ax.plot(x, y, color="green", label="Entrenamiento")
    ax.annotate(text=f"MSE={crit(pars):2f}", xy=[-3,0.75])
    fig.suptitle("Red Neuronal vs Entrenamiento")
    ax.legend()