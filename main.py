from Perceptron import Perceptron, train, f
from Layer import Layer
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def generate_data(batch_size):
    X = np.sort(np.random.uniform(-10, 10, batch_size)).reshape(1, -1)
    Y = f(X)
    X = X / 10
    return X, Y


def visualize_loss(epochs, losses):
    plt.plot(range(epochs), losses, label="Strata")
    plt.legend()
    plt.title("Strata w zależności od epoki")
    plt.xlabel("epoka")
    plt.ylabel("strata")
    plt.grid(True)
    plt.show()


def visualize_functions(X, Y_pred, Y_true):
    plt.plot(X.flatten(), Y_true.flatten(), label="f(x)")
    plt.plot(X.flatten(), Y_pred.flatten(), label="h(x)")
    plt.legend()
    plt.title("Rzeczywiste wartości funkcji f(x) oraz aproksymowane h(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def visualize_loss_over_layer_size(layer_sizes, losses):
    plt.plot(layer_sizes, losses, label="Strata")
    plt.legend()
    plt.title("Strata w zależności od liczby warstwach ukrytych")
    plt.xlabel("liczba warstw ukrytych")
    plt.ylabel("strata")
    plt.grid(True)
    plt.show()


def visualize_time_over_layer_size(layer_sizes, losses):
    plt.plot(layer_sizes, losses, label="Czas")
    plt.legend()
    plt.title("Czas treningu sieci w zależności od liczby warstw ukrytych")
    plt.xlabel("liczba warstw ukrytych")
    plt.ylabel("czas [s]")
    plt.grid(True)
    plt.show()


def test(epochs=100, batch_size=100):
    X_train, Y_train = generate_data(batch_size)
    X_test, Y_test = generate_data(batch_size)
    model = Perceptron([1, 50, 50, 50, 50, 50, 1])
    losses = train(model, X_train, Y_train, epochs, batch_size, print_status=True)
    visualize_loss(epochs, losses)
    Y_pred = model.predict(X_test)
    visualize_functions(X_test, Y_pred, Y_test)


def test_layer_size(X_train, Y_train, epochs=100, batch_size=100, max_layer_size=100):
    layer_sizes = range(1, max_layer_size + 1, 10)
    all_losses = []
    times = []
    for i in layer_sizes:
        model = Perceptron([1, i, i, i, i, 1])
        start = time.time()
        losses = train(model, X_train, Y_train, epochs, batch_size, print_status=True)
        end = time.time()
        times.append(end - start)
        all_losses.append(losses[-1])
        # visualize_loss(epochs, losses)
        # Y_pred = model.predict(X_test)
        # visualize_functions(X_test, Y_pred, Y_test)
    visualize_loss_over_layer_size(layer_sizes, all_losses)
    visualize_time_over_layer_size(layer_sizes, times)


def test_layer_count(X_train, Y_train, epochs=100, batch_size=100, max_layer_count=100):
    layer_counts = range(1, max_layer_count + 1)
    all_losses = []
    times = []
    layer_count_table = [1, 1]
    for i in layer_counts:
        print(f"Hidden layers: {i}")
        layer_count_table.insert(-1, 50)
        model = Perceptron(layer_count_table)
        start = time.time()
        losses = train(model, X_train, Y_train, epochs, batch_size, print_status=True)
        end = time.time()
        times.append(end - start)
        all_losses.append(losses[-1])
    visualize_loss_over_layer_size(layer_counts, all_losses)
    visualize_time_over_layer_size(layer_counts, times)


def test_all(X_train, Y_train, X_test, Y_test, epochs=100, batch_size=100):
    model = Perceptron([1, 50, 50, 50, 50, 50, 1])
    start = time.time()
    losses = train(model, X_train, Y_train, epochs, batch_size, print_status=False)
    end = time.time()
    print(f"Time MLP: {end-start} [s]")
    Y_pred_perceptron = model.predict(X_test)
    visualize_loss(epochs, losses)

    model_poly_regr = make_pipeline(PolynomialFeatures(degree=35), LinearRegression())
    start = time.time()
    model_poly_regr.fit(X_train.T, Y_train.T)
    end = time.time()
    print(f"Time polynomial regression: {end-start} [s]")
    Y_pred_poly_regr = model_poly_regr.predict(X_test.T)
    error_regr = np.mean((Y_pred_poly_regr.T - Y_test) ** 2)
    error_mlp = np.mean((Y_pred_perceptron - Y_test) ** 2)
    print(f"Error MLP: {error_mlp} [s]")
    print(f"Error regression: {error_regr} [s]")
    print(Y_pred_poly_regr.T.shape)
    print(Y_test.shape)
    visualize_functions_3(10 * X_test, Y_pred_perceptron, Y_pred_poly_regr, Y_test)


def visualize_functions_3(X, Y_pred1, Y_pred2, Y_true):
    plt.plot(
        X.flatten(), Y_true.flatten(), label="f(x) - wyjściowa funkcja", color="red"
    )
    plt.plot(
        X.flatten(),
        Y_pred1.flatten(),
        label="h(x) - perceptron wielowarstwowy",
        color="orange",
    )
    plt.plot(
        X.flatten(),
        Y_pred2.flatten(),
        label="g(x) - regresja wielomianowa",
        color="green",
    )
    plt.legend()
    plt.title("Rzeczywiste wartości funkcji f(x) oraz aproksymowane g(x) i h(x) ")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    X_train, Y_train = generate_data(batch_size=1000)
    X_test, Y_test = generate_data(batch_size=1000)
    # test(epochs = 2000, batch_size = 1000)
    # test_layer_size( X_train, Y_train, epochs=2000, batch_size=1000, max_layer_size=300)
    # test_layer_count(X_train, Y_train, epochs=2000, batch_size=1000, max_layer_count=20)
    test_all(X_train, Y_train, X_test, Y_test, epochs=2000, batch_size=1000)
