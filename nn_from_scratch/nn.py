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
    'relu': (lambda x: np.max(0, x), lambda x: 0 if x < 0 else 1),
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
    def __init__(self, learning_rate, loss_functions, n_iter, optimizer='gradient_descent'):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.n_iter = n_iter
    
    def optimize(self, backprop_dict, loss_functions, activation_functions):
        loss_function = LossFunction(loss_function)
        if self.optimizer == 'gradient_descent':
            for iter in range(self.n_iter):
                i = len(self.layers)
                a = backprop_dict[f'deriv_layer_{i}'][f'a_{i}']
                z = backprop_dict[f'deriv_layer_{i}'][f'z_{i}']
                delta = loss_function.apply_derivative(a_L, y) * activation_function.apply_derivative(z)
                for i in range(len(self.layers)):
                    W -= self.learning_rate * (delta @ a_prev)
                    b -= self.learning_rate * delta
                    a = backprop_dict[f'deriv_layer_{i}'][f'a_{i-1}']
                    z = backprop_dict[f'deriv_layer_{i}'][f'z_{i-1}']
                    delta = self.layers[i].W.T @ delta * activation_function.apply_derivative(z)
                    self.layers[i].W = W
                    self.layers[i].b = b


class Layer():
    def __init__(self, input, units, activation='linear'):
        self.n_x = input.shape[0]
        self.units = units
        self.W = np.random((self.units, self.n_x))
        self.b = np.random((self.units, 1))
        self.activation = activation
    

class FFNeuralNetwork():
    def __init__(self, layers, inputs, labels):
        self.layers = layers
        self.inputs = inputs
        self.labels = labels
    
    def initialize(self):
        pass

    def train(self, epochs, optimizer):
        # Create a dictionnary to store each values needed for the backpropagation
        backprop_dict = {f'deriv_layer_{i}': {} for i in range(len(self.layers))}
        a_i = np.copy(self.inputs)
        for epoch in range(epochs):
            for i in range(len(self.layers)):
                # Perform the forward propagation through each layer
                # and stores each value computed
                backprop_dict[f'deriv_layer_{i}'][f'a_{i - 1}'] = a_i
                z = self.layers[i].W @ a_i + self.layers[i].b
                a = self.layers[i].activation(z)
                a_i = a
                backprop_dict[f'deriv_layer_{i}'][f'z_{i}'] = a_i
                backprop_dict[f'deriv_layer_{i}'][f'W_{i}'] = self.layers[i].W
                #backprop_dict[f'deriv_layer_{i}'][f'b_{i}'] = self.layers[i].b

            optimizer.optimize(backprop_dict, functions)
