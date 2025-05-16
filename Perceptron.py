import numpy as np
import math
from Layer import Layer


class Perceptron:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, train_data):
        X = train_data
        for layer in self.layers[:-1]:
            X = layer.forward(X)
        self.layers[-1].z = self.layers[-1].W @ X + self.layers[-1].b
        self.layers[-1].a = self.layers[-1].z
        return self.layers[-1].z

    def backward(self, X, Y, gamma):
        Y_pred = self.forward(X)
        L = loss_function(Y_pred, Y)
        dL_dZ = 2 * (Y_pred - Y) / Y.shape[1]
        dA = dL_dZ
        dL_dW = dA @ self.layers[-2].a.T
        dL_db = np.sum(dA, axis=1, keepdims=True)
        dA = self.layers[-1].W.T @ dA
        self.layers[-1].W -= gamma * dL_dW
        self.layers[-1].b -= gamma * dL_db
        for i in reversed(range(len(self.layers) - 1)):
            dA = self.layers[i].backward(dA, gamma)
        return L

    def predict(self, X):
        return self.forward(X)


def f(x):
    return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


def loss_function(Y_pred, Y):
    return np.mean((Y_pred - Y) ** 2)


def train(model, X, Y, epochs, batch_size, print_status=True):
    losses = []
    for i in range(1, epochs + 1):
        loss = model.backward(X, Y, 0.01)
        losses.append(loss)
        if print_status:
            print("Epoch: {}, Loss: {}".format(i, loss))
    return losses
