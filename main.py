from Perceptron import Perceptron
from Layer import Layer
import numpy as np

def test():
    # layer1 = Layer(1,4)
    # print(layer1.W)
    X = np.array([[1]])
    # forwarded = layer1.forward(X)
    # print(X)
    # print(layer1.a)
    p = Perceptron([1,4,4,1])
    print(p.forward(X))

if __name__ == '__main__':
    test()
