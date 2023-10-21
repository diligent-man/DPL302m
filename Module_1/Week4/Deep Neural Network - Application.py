import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *


##############################################################################################################
def dataset_inspecting(train_x: np.ndarray, train_y: np.ndarray,
                       test_x: np.ndarray, test_y: np.ndarray, classes: list) -> None:
    m_train = train_x.shape[0]
    num_px = train_x.shape[1]
    m_test = test_x.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x shape: " + str(train_x.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x shape: " + str(test_x.shape))
    print ("test_y shape: " + str(test_y.shape))
    return None


def dataset_preprocessing(train_x: np.ndarray, test_x: np.ndarray, classes: list) -> tuple:
    # flatten img
    train_x = train_x.reshape(train_x.shape[0], -1).T
    test_x = test_x.reshape(test_x.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x / 255
    test_x = test_x / 255

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    # convert binary -> str
    classes = np.char.decode(classes, encoding="utf-8")
    return train_x, test_x, classes

##############################################################################################################
'''
1/ The input is a (64,64,3) image which is flattened to a vector of size  (12288,1)
.
2/ The corresponding vector:  [𝑥0,𝑥1,...,𝑥12287]𝑇 is then multiplied by
the weight matrix  𝑊[1] of size  (𝑛[1],12288)
.
3/ Then, add a bias & take its relu to get the following vector:  [𝑎[1]0,𝑎[1]1,...,𝑎[1]𝑛[1]−1]𝑇
.
4/ Multiply the resulting vector by  𝑊[2] and add bias.

5/ Finally, take the sigmoid of the result. If it's greater than 0.5, classify it as a cat.
'''
def two_layer_model(X: np.ndarray, Y: np.ndarray, layers_dims: list,
                    learning_rate=0.0075, num_iterations=3000, print_cost=False) -> tuple:
    np.random.seed(1)
    gradients = {}
    costs = []                             
    m = X.shape[1]
    inp_size, hidden_size, output_size = layers_dims
    parameters = initialize_parameters(inp_size, hidden_size, output_size)
    
    # Retrieve paras
    W1 = parameters["W1"]; b1 = parameters["b1"]
    W2 = parameters["W2"]; b2 = parameters["b2"]
    
    # Gradient descent
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        
        cost = compute_cost(A2, Y)
        
        # Back prop init
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Back prop
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        # Gradient descents
        gradients['dW1'] = dW1; gradients['db1'] = db1
        gradients['dW2'] = dW2; gradients['db2'] = db2
        
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]; b1 = parameters["b1"]
        W2 = parameters["W2"]; b2 = parameters["b2"]
        
        # Print the cost every 100 iterations
        if print_cost and (i % 100 == 0) or (i == num_iterations-1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    return parameters, costs


def plot_costs(costs, learning_rate=0.0075) -> None:
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return None

def two_layer_training(train_x: np.ndarray, train_y: np.ndarray,
                       test_x: np.ndarray, test_y: np.ndarray, classes: list) -> None:
    # Init
    inp_size = 12288
    hidden_size = 7
    output_size = 1
    layers_dims = (inp_size, hidden_size, output_size)
    learning_rate = 0.0075

    # Training
    parameters, costs = two_layer_model(train_x, train_y, layers_dims=layers_dims,
                                        num_iterations=2500, print_cost=True)
    plot_costs(costs, learning_rate)

    # Model evaluation
    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)
    return None


##############################################################################################################
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    
    # Gradient descent
    for i in range(0, num_iterations):
        # Forward prop: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)

        # Backward prop
        grads = L_model_backward(AL, Y, caches)
        
        # Update paras
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    return parameters, costs


def L_layer_training(train_x: np.ndarray, train_y: np.ndarray,
                       test_x: np.ndarray, test_y: np.ndarray, classes: list) -> None:
    # Init
    inp_size = 12288
    hidden_size = 7
    output_size = 1
    layers_dims = (inp_size, hidden_size, output_size)
    learning_rate = 0.0075
    L_layer_model()

    return None

##############################################################################################################
def main() -> None:
    train_x, train_y, test_x, test_y, classes = load_data()
    dataset_inspecting(train_x, train_y, test_x, test_y, classes)
    train_x, test_x, classes = dataset_preprocessing(train_x, test_x, classes)
    

    two_layer_training(train_x, train_y, test_x, test_y, classes)
    L_layer_training(train_x, train_y, test_x, test_y, classes)
    return None


if __name__ == '__main__':
    main()

