import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec


def model(X: np.ndarray, Y: np.ndarray, learning_rate=0.01,
          num_iterations=15000, print_cost=True, initialization="he"):
    """
    NN arch: RELU(LINEAR) -> RELU(LINEAR) -> SIGMOID(LINEAR)
    Shape: X=(2, # of ex)
           Y=(1, # of ex)
    """

    grads = {}
    costs = []  # to keep track of the loss
    m = X.shape[1]  # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)
    for i in range(num_iterations):
        # Forward prop
        a3, cache = forward_propagation(X, parameters)
        # Loss
        cost = compute_loss(a3, Y)
        # Backprop
        grads = backward_propagation(X, Y, cache)
        # Update params
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


###########################################################################################
# Zero init
def initialize_parameters_zeros(layers_dims: list):
    paras = {}
    L = len(layers_dims)
    for l in range(1, L):
        paras['W' + str(l)] = np.zeros(shape=(layers_dims[l], layers_dims[l-1]))
        paras['b' + str(l)] = np.zeros(shape=(layers_dims[l], 1))
    return paras

def Exercise_1():
    parameters = initialize_parameters_zeros([3, 2, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    return None

def model_with_zero_init(x_train: np.ndarray, y_train: np.ndarray,
                         x_test: np.ndarray, y_test: np.ndarray):
    paras = model(x_train, y_train,initialization='zeros', num_iterations=15000)
    print("On the train set:")
    predictions_train = predict(x_train, y_train, paras)
    print("On the test set:")
    predictions_test = predict(x_test, y_test, paras)

    # visualize
    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(paras, x.T), x_train, y_train)

    print("predictions_train = " + str(predictions_train))
    print("predictions_test = " + str(predictions_test))

    # remark
    # model yields 0 because weights and biases are not updated
    # Hence, Weight should be arbitrarily initialized, biases=0 is ok
    return None


##########################################################################################
# Random init
def initialize_parameters_random(layers_dims: list):
    paras = {}
    L = len(layers_dims)
    for l in range(1, L):
        paras['W' + str(l)] = np.random.rand(layers_dims[l], layers_dims[l-1]) * layers_dims[l-1]
        paras['b' + str(l)] = np.zeros(shape=(layers_dims[l], 1))
    return paras

def Exercise_2():
    parameters = initialize_parameters_random([3, 2, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    return None


def model_with_random_init(x_train: np.ndarray, y_train: np.ndarray,
                         x_test: np.ndarray, y_test: np.ndarray):
    paras = model(x_train, y_train,initialization='random', num_iterations=150000)
    print("On the train set:")
    predictions_train = predict(x_train, y_train, paras)
    print("On the test set:")
    predictions_test = predict(x_test, y_test, paras)

    # visualize
    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(paras, x.T), x_train, y_train)

    print("predictions_train = " + str(predictions_train))
    print("predictions_test = " + str(predictions_test))

    # Remark
    # Initializing weights to very large random values doesn't work well.
    # Initializing with small random values should do better.
    # The important question is, how small should be these random values be?
    return None


########################################################################################################
# he init
def initialize_parameters_he(layers_dims: list):
    paras = {}
    L = len(layers_dims)

    for l in range(1, L):
        scaling_factor = 2 / np.sqrt(layers_dims[l-1])
        paras['W' + str(l)] = np.random.rand(layers_dims[l], layers_dims[l-1]) * scaling_factor
        paras['b' + str(l)] = np.zeros(shape=(layers_dims[l], 1))
    return paras

def Exercise_3():
    parameters = initialize_parameters_he([3, 2, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    return None


def model_with_he_init(x_train: np.ndarray, y_train: np.ndarray,
                         x_test: np.ndarray, y_test: np.ndarray):
    paras = model(x_train, y_train, initialization='he', num_iterations=150000)
    print("On the train set:")
    predictions_train = predict(x_train, y_train, paras)
    print("On the test set:")
    predictions_test = predict(x_test, y_test, paras)

    # visualize
    plt.title("Model with he initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(paras, x.T), x_train, y_train)

    print("predictions_train = " + str(predictions_train))
    print("predictions_test = " + str(predictions_test))

    # Remark:
    # Initializing weights to very large random values doesn't work well.
    # Initializing with small random values should do better.
    # The important question is, how small should be these random values be?
    return None


########################################################################################################
def main() -> None:
    x_train, y_train, x_test, y_test = load_dataset()
    # Exercise_1()
    # model_with_zero_init(x_train, y_train, x_test, y_test)
    # Exercise_2()
    # model_with_random_init(x_train, y_train, x_test, y_test)

    Exercise_3()
    model_with_he_init(x_train, y_train, x_test, y_test)
    return None

if __name__ == '__main__':
    main()