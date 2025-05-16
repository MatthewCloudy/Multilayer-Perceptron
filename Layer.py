import numpy as np

class Layer:
    def __init__(self, in_size, layer_size):
        self.W = np.random.randn(layer_size, in_size).astype(np.float32)
        self.b = np.random.randn(layer_size, 1).astype(np.float32)
        self.z = None
        self.a = None
        self.X = None


    def forward(self, X):
        self.X = X
        self.z = self.W @ X + self.b
        self.a = np.tanh(self.z)
        return self.a


    def backward(self, dA, gamma):
        dL_dZ = dA * activation_derivative(self.z)
        dL_dW = dL_dZ @ self.X.T / self.X.shape[1]
        dL_db = np.sum(dL_dZ, axis=1, keepdims=True) / self.X.shape[1]

        dA_prev = self.W.T @ dL_dZ

        self.W -= gamma * dL_dW
        self.b -= gamma * dL_db

        return dA_prev

def activation_derivative(z):
    return 1 - np.tanh(z)**2