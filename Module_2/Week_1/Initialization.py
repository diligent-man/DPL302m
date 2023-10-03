import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec


### Neural NetWork Model

def model(X: np.ndarray, Y: np.ndarray, learning_rate = 0.01, num_iterations = 15000,
          print_cost = True, initialization = "he"):
    """"
    3-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    shape:  X(2, ex)
            Y(1, ex)
    """
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == "zeros": # tất cả các trọng số và bias ban đầu đều được thiết lập thành 0.
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":#phương pháp này làm cho các trọng số và bias có giá trị ngẫu nhiên,
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":#tránh vấn đề gradient biến mất (vanishing gradient) trong quá trình huấn luyện mạng nơ-ron sâu
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(num_iterations):
        #LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        #Loss
        cost = compute_loss(a3, Y)
        #Backprop
        grads = backward_propagation(X, Y, cache)
        # Update params.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

############################################################

### Zero Initialization
def initialize_parameters_zeros(layers_dims: list):
    paras = {}
    L = len(layers_dims)
    for l in range(1, L):
        paras['W' + str(l)] = np.zeros(shape=(layers_dims[l],layers_dims[l-1]))
        paras['b' + str(l)] = np.zeros(shape=(layers_dims[l],1))
    return paras

def Exercise_1():
    parameters = initialize_parameters_zeros([3, 2, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    return None

## Model on 15,000 iterations using zeros initialization.
def model_with_zero_init(train_X: np.ndarray, train_Y: np.ndarray,
                         test_X: np.ndarray, test_Y: np.ndarray):
    paras = model(train_X, train_Y,initialization='zeros', num_iterations=15000)
    print("On the train set:")
    predictions_train = predict(test_X, train_Y, paras)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, paras)

## Visualize model with Zeros initialization
    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(paras, x.T), train_X, train_Y)

    print("predictions_train = " + str(predictions_train))
    print("predictions_test = " + str(predictions_test))

    return None


### Random init
def initialize_parameters_random(layers_dims: list):
    paras = {}
    L = len(layers_dims)
    for l in range(1, L ):
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

## Model on 15,000 iterations using random initialization.
def model_with_random_init(train_X: np.ndarray, train_Y: np.ndarray,
                         test_X: np.ndarray, test_Y: np.ndarray):
    paras = model(train_X, train_Y, initialization="random", num_iterations=15000)
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, paras)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, paras)

## Visualize model with random initialization.
    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(paras, x.T), train_X, train_Y)

### He initialization
def initialize_parameters_he(layers_dims):
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

## Model on 15,000 iterations using He initialization.
def model_with_he_init(train_X: np.ndarray, train_Y: np.ndarray,
                         test_X: np.ndarray, test_Y: np.ndarray):
    paras = model(train_X, train_Y, initialization='he', num_iterations=15000)
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, paras)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, paras)

## Visualize model with He initialization.
    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(paras, x.T), train_X, train_Y)

    print("predictions_train = " + str(predictions_train))
    print("predictions_test = " + str(predictions_test))





#########################################################################################
def main() -> None:
    train_X, train_Y, test_X, test_Y = load_dataset()
    # Exercise_1()
    # model_with_zero_init(train_X, train_Y, test_X, test_Y)
    # Exercise_2()
    # model_with_random_init(train_X, train_Y, test_X, test_Y)

    Exercise_3()
    model_with_he_init(train_X, train_Y, test_X, test_Y)
    return None

if __name__ == '__main__':
    main()