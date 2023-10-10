import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters


def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]
    parameters = initialize_parameters(layers_dims)
    # GD
    for i in range(0, num_iterations):
        # Forward prop
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        # Cost func
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
        # Backprop
        assert (lambd == 0 or keep_prob == 1)
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        parameters = update_parameters(parameters, grads, learning_rate) # Update parameters
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

def model_without_regularization(x_train, y_train, x_test, y_test) -> None:
    parameters = model(x_train, y_train, num_iterations=600000)
    print("On the training set:")
    predictions_train = predict(x_train, y_train, parameters)
    print("On the test set:")
    predictions_test = predict(x_test, y_test, parameters)

    # visualize
    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), x_train, y_train)
    return None
#########################################################################################
# Exercise 1 - compute_cost_with_regularization
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    cross_entropy_cost = compute_cost(A3, Y) # compute cross entropy
    L2_regularization_cost = (1 / m) * (lambd / 2) * np.sum([np.sum(np.square(Wl)) for Wl in [W1, W2, W3]])
    cost = cross_entropy_cost + L2_regularization_cost
    return cost

###########################################################################################
# Exercise 2 - backward_propagation_with_regularization
def backward_propagation_with_regularization(X: np.ndarray, Y: np.ndarray, cache: dict, lambd: float) -> dict:
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    # third layer
    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T) + (lambd * W3) / m
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    # second layer
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) + (lambd * W2) / m
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    # first layer
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T) + (lambd * W1) / m
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

def model_with_L2_regularization(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    parameters = model(x_train, y_train, num_iterations=100000, lambd=.05)
    print("On the training set:")
    predictions_train = predict(x_train, y_train, parameters)
    print("On the test set:")
    predictions_test = predict(x_test, y_test, parameters)

    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), x_train, y_train)
    return None

##########################################################################################
# Exercise 3: - forward_propagation_with_dropout
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    np.random.seed(1)
    W1 = parameters["W1"]; b1 = parameters["b1"]
    W2 = parameters["W2"]; b2 = parameters["b2"]
    W3 = parameters["W3"]; b3 = parameters["b3"]
    # First layer
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 = A1 / keep_prob
    # second layer
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob
    # third layer
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache

#########################################################################################
# Exercise 4 - backward_propagation_with_dropout
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3 = cache
    # third layer
    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 *= D2
    dA2 /= keep_prob
    # second layer
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 *= D1
    dA1 /= keep_prob
    # first layer
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

def model_with_dropout(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, y_test: np.ndarray) -> None:
    parameters = model(x_train, y_train, keep_prob = 0.86, learning_rate = 0.05, num_iterations=100_000)
    print("On the training set:")
    predictions_train = predict(x_train, y_train, parameters)
    print("On the test set:")
    predictions_test = predict(x_test, y_test, parameters)

    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), x_train, y_train)
    return None

#########################################################################################
def main() -> None:
    x_train, y_train, x_test, y_test = load_2D_dataset()
    model_without_regularization(x_train, y_train, x_test, y_test)
    model_with_L2_regularization(x_train, y_train, x_test, y_test)
    model_with_dropout(x_train, y_train, x_test, y_test)
    return None


if __name__ == '__main__':
    main()