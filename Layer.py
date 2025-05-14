import numpy as np

class Layer:
    def __init__(self, in_size, layer_size):
        self.W = np.random.randn(layer_size, in_size).astype(np.float32)
        self.b = np.random.randn(layer_size, 1).astype(np.float32)
        self.s = None
        self.a = None


    def forward(self, X):
        self.s = self.W @ X + self.b
        self.a = np.tanh(self.s)
        return self.a


    def backward(self, dA):
        self.s = dA
