from neuralnetworks.maths import gradient
import numpy as np
import matplotlib.pyplot as plt


# Funciones de Activación
def sigmoid(x): 
    # Función Sigmoide
    y = 1 / (1 + np.exp(-x))
    return y


# Algoritmos de Oprimización
def descent_gradent(func, theta0, epsilon, lr = 0.01):
    """
        Algoritmo del Gradiente Descendiente
        func: Función a minimizar
        theta0: Parámetro inicial
        epsilon: Error
        lr: Tasa de Arendizaje
    """
    error = func(theta0)
    epochs = 0
    while error > epsilon:
        thetan = theta0 - lr * gradient(func, theta0)
        error = func(thetan)
        t_n1 = np.array(thetan)
        t_n = np.array(theta0)
        diff = (t_n1 - t_n) ** 2
        near = np.sqrt(diff.sum())
        if near < epsilon:
            break
        theta0 = thetan
        if epochs % 1000 == 0:
            print(epochs, near)
        epochs+=1
    print("Error", error)
    return theta0
