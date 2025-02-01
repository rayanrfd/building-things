import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(- x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


lf = {
    'mse': (lambda y, y_hat : np.mean(np.linalg.norm(y - y_hat)), 
            lambda y, y_hat : (2/y.shape[0]) * (y - y_hat)),
    'binary_crossentropy': (lambda y, y_hat : - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)),
                            lambda y, y_hat : - ((y / y_hat) + ((1 - y) / (1 - y_hat)))),
}

af = {
    'linear': (lambda x: x, lambda x: 1),
    'relu': (lambda x: np.max(0, x)),
    'sigmoid': (sigmoid, d_sigmoid)
}


class ActivationFunction():
    def __init__(self, name):
        self.name = name
    
    def apply(self):
        return af[self.name][0]
    
    def apply_derivative(self):
        return af[self.name][1]


class LossFunction():
    def __init__(self, name):
        self.name = name
    
    def apply(self):
        return lf[self.name][0]
    
    def apply_derivative(self):
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
                pass


class Layer():
    def __init__(self, input, units, activation='linear'):
        self.n_x = input.shape[0]
        self.units = units
        self.weights = np.random((self.units, self.n_x))
        self.biases = np.random((self.units, 1))
        self.activation = activation
    

class FFNeuralNetwork():
    def __init__(self, layers, inputs, labels, optimizer):
        self.layers = layers
        self.inputs = inputs
        self.labels = labels
        self.optimizer = optimizer
    
    def initialize(self):
        pass

    def train(self, epochs):
        inputs = np.copy(self.inputs)
        for epoch in range(epochs):
            for i in range(len(self.layers)):
                z = self.layers[i].weights @ self.inputs + self.layers[i].biases
                a = self.layers[i].activation(z)
                inputs = a
