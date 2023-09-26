import copy
import h5py
import numpy as np
import matplotlib.pyplot as plt


############################################################################################
# Built-in funcs
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):    
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ


############################################################################################
# Exercise_1
# Init paras for 2-layer NN
def initialize_parameters(inp_size: int, hidden_size: int, output_size: int) -> dict:
    np.random.seed(1)
    
    W1 = np.random.randn(hidden_size, inp_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


#############################################################################################################
# Exercise_2
# Init paras for the whole DNN
def initialize_parameters_deep(layer_dims: list) -> dict:
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for layer in range(1, L):
        current_layer_size = layer_dims[layer]
        prev_layer_size = layer_dims[layer-1]
        parameters[f"W{layer}"] = np.random.randn(current_layer_size, prev_layer_size) * 0.01
        parameters[f"b{layer}"] = np.zeros((current_layer_size, 1))
         
        assert(parameters['W' + str(layer)].shape == (current_layer_size, prev_layer_size))
        assert(parameters['b' + str(layer)].shape == (current_layer_size, 1))
    return parameters


#################################################################################################
# Exercse 3
def linear_forward(W: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple:
    Z = W @ A + b
    cache = (A, W, b)
    return Z, cache


################################################################################################
# Exercise 4
def linear_activation_forward(W: np.ndarray, A_prev: np.ndarray, b: np.ndarray, activation: str) -> tuple:
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(W, A_prev, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(W, A_prev, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


#################################################################################################
# Exercise 5
def L_model_forward(X: np.ndarray, parameters: dict) -> tuple:
    caches = []
    A = X
    # number of layers in the neural network
    L = len(parameters) // 2                  
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(parameters[f"W{l}"], A_prev, , parameters[f"b{l}"], "relu")
        caches.append(cache)
    
    # LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    # AL: y_hat
    AL, cache = linear_activation_forward(parameters[f"W{L}"], A, parameters[f"b{L}"], "sigmoid")
    caches.append(cache) 
    return AL, caches


#################################################################################################
# Exercise 6
# cost of cross entropy
def compute_cost(AL, Y):
    # Compute loss from AL and y.
    m = Y.shape[1]
    cost = -1 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)    
    return cost


#################################################################################################

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * (dZ @ A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ
    return dA_prev, dW, db



def main() -> None:
    
    return None







if __name__ == '__main__':
    np.random.seed(1)
    plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    main()