import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(- x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


lf = {
    'mse': (lambda y, y_hat : np.mean((y - y_hat) ** 2), 
            lambda y, y_hat : (2/y.shape[0]) * (y - y_hat)),
    'binary_crossentropy': (lambda y, y_hat : - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)),
                            lambda y, y_hat : - ((y / y_hat) - ((1 - y) / (1 - y_hat)))),
}

af = {
    'linear': (lambda x: x, lambda x: 1),
    'relu': (lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0)),
    'sigmoid': (sigmoid, d_sigmoid)
}


class ActivationFunction():
    def __init__(self, name):
        self.func, self.deriv = af[name]


class LossFunction():
    def __init__(self, name):
        self.func, self.deriv = lf[name]
    

class Optimizer():
    def __init__(self, learning_rate, loss_function: LossFunction, n_iter, optimizer='gradient_descent'):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.n_iter = n_iter
    
    def optimize(self, backprop_list, layers, activation_functions, labels):
        if self.optimizer == 'gradient_descent':
            for iter in range(self.n_iter):
                i = len(layers)
                a = backprop_list[i]['a']
                z = backprop_list[i]['z']
                a_prev = backprop_list[i]['a_prev']
                delta = self.loss_function.deriv(a, labels) * activation_functions[i].deriv(z)
                for i in range(len(layers) - 1, -1, -1):
                    d_W = self.learning_rate * np.dot(delta, a_prev)
                    d_b = self.learning_rate * delta
                    a = backprop_list[i-1]['a']
                    z = backprop_list[i-1]['a']
                    delta = layers[i].W.T @ delta * activation_functions[i].deriv(z)
                    layers[i].W -= d_W
                    layers[i].b -= d_b


class Layer():
    def __init__(self, input_size, units, activation: ActivationFunction):
        self.n_x = input_size.shape[0]
        self.units = units
        self.W = np.random.randn(self.units, self.n_x)
        self.b = np.zeros(self.units, 1)
        self.activation = activation
    

class FFNeuralNetwork():
    def __init__(self, layers, inputs, labels):
        self.layers = layers
        self.inputs = inputs
        self.labels = labels

    def train(self, epochs, optimizer):
        activation_functions = [self.layers[i].activation for i in range(len(self.layers))]
        a_i = np.copy(self.inputs)
        for epoch in range(epochs):
            # Create a list to store each values needed for the backpropagation
            backprop_list = []
            for i in range(len(self.layers)):
                # Perform the forward propagation through each layer
                # and stores each value computed in a dictionnary
                a_prev = a_i
                z = self.layers[i].W @ a_i + self.layers[i].b
                a = activation_functions[i].func(z)
                a_i = a
                backprop_list.append({'a_prev': a_prev, 'z': z, 'a': a, 'layer': i})

            optimizer.optimize(backprop_list, self.layers, activation_functions, self.labels)
    
    def inference(self, inputs):
        a_i = np.copy(inputs)
        for i in range(len(self.layers)):
            z = self.layers[i].W @ a_i + self.layers[i].b
            a = self.layers[i].activation.func(z)
            a_i = a
        return a_i
