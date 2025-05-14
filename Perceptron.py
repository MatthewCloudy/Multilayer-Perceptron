import numpy as np
import math
from Layer import Layer

class Perceptron:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, train_data):
        X = train_data
        for layer in self.layers[:-1]:
            X = layer.forward(X)
        self.layers[-1].s = self.layers[-1].W @ X + self.layers[-1].b
        self.layers[-1].a = self.layers[-1].s
        return self.layers[-1].s

    def backward(self, X, Y):
        Y_pred = self.forward(X)
        loss = loss_function(Y_pred, Y)
        gradient = 2*(Y_pred - Y) / Y.shape[1]



    def predict(self, X):
        return self.forward(X)

    # def __str__(self):
    #     string = ""
    #     for layer in self.layers:
    #         string += layer.W + "\n"

def f(x):
    return x**2*np.sin(x) + 100*np.sin(x)*np.cos(x)

def loss_function(Y_pred, Y)
    return np.mean((Y_pred - Y)**2)