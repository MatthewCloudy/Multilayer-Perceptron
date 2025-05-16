from Perceptron import Perceptron, train, f
from Layer import Layer
import numpy as np
from matplotlib import pyplot as plt


def generate_data(batch_size):
    np.random.seed(42)
    X = np.sort(np.random.uniform(-10, 10, batch_size)).reshape(1, -1)
    Y = f(X)
    return X, Y

def visualize_loss(epochs, losses):
    plt.plot(range(epochs), losses, label='Strata')
    plt.legend()
    plt.title("Strata w zależności od epoki")
    plt.grid(True)
    plt.show()

def visualize_functions(X, Y_pred, Y_true):
    plt.plot(X.flatten(), Y_true.flatten(), label='f(x)')
    plt.plot(X.flatten(), Y_pred.flatten(), label='f_pred(x)')
    plt.legend()
    plt.title("Rzeczywiste wartosci funkcji i z regresji")
    plt.grid(True)
    plt.show()

def test(epochs = 100, batch_size=100):
    X_train, Y_train = generate_data(batch_size)
    X_test, Y_test = generate_data(batch_size)
    model = Perceptron([1,64,64,1])
    losses = train(model, X_train, Y_train, epochs, batch_size, print_status=True)
    visualize_loss(epochs, losses)
    Y_pred = model.predict(X_test)
    visualize_functions(X_test,Y_pred,Y_test)

if __name__ == '__main__':
    test(epochs = 10000, batch_size = 1000)
