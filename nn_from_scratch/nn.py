import numpy as np


lf = {
    'mse': (lambda y, y_hat : np.linalg.norm(y - y_hat), 
            lambda y, y_hat : 2 * (y - y_hat)),
    'binary_crossentropy': (lambda y, y_hat : - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)),
                            lambda y, y_hat : - ((y / np.log(y_hat)) + ((1 - y) / np.log(1 - y_hat)))),
}

af = {
    'linear': (lambda x: x, lambda x: 1),
    'relu': (lambda x: np.max(0, x)),
    'sigmoid': (lambda x: 1 / (1 + np.exp(- x)), lambda x: 1 / (1 + np.exp(- x)) * (1 - (1 / (1 + np.exp(- x)))))
}


class ActivationFunction():
    def __init__(self, name):
        self.name = name
    
    def compute(self):
        return af[self.name][0]
    
    def compute_derivative(self):
        return af[self.name][1]


class LossFunction():
    def __init__(self, name):
        self.name = name
    
    def compute(self):
        return lf[self.name][0]
    
    def compute_derivative(self):
        return lf[self.name][1]
    

class Optimizer():
    def __init__(self, learning_rate, weights, biases, loss_function, n_iter, optimizer='gradient_descent'):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weights = weights
        self.biases = biases
        self.loss_function = loss_function
        self.n_iter = n_iter
    
    def optimize(self, weights, biases, loss_function, n_iter, optimizer='gradient_descent'):
        loss_function = LossFunction(loss_function)
        if optimizer == 'gradient_descent':
            for iter in range(n_iter):



class Layer():
    def __init__(self, input, units, activation):
        self.n_x = input.shape[1]
        self.units = units
        self.weights = np.random((self.units, self.n_x))
        self.biases = np.random((self.units, 1))
        #self.activation = activation
    

class FFNeuralNetwork():
    def __init__(self, layers):
        self.layers = layers
    
    def train(self, epochs):
        for epoch in range(epochs):
            pass
        