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
def compute_cost(AL: np.ndarray, Y: np.ndarray) -> float:
    # Compute loss from AL and y.
    m = Y.shape[1]
    cost = -1 / m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)    
    return cost


#################################################################################################
# Exercise 7
def linear_backward(dZ: np.ndarray, cache: dict) -> tuple:
    W, A_prev, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * (dZ @ A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ
    return dA_prev, dW, db


#################################################################################################
# Exercise 8
def linear_activation_backward(dA: np.ndarray, cache: dict, activation: np.ndarray) -> tuple:
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dW, dA_prev, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dW, dA_prev, db = linear_backward(dZ, linear_cache)        

    return dA_prev, dW, db


#################################################################################################
#Exercise 9
def L_model_backward(AL: np.ndarray, Y: np.ndarray, caches: dict) -> dict:
    gradients = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients
    dW_temp, dA_prev_temp, db_temp = linear_activation_backward(dAL, caches[L-1], "sigmoid")
    gradients[f"dA{L-1}"] = dA_prev_temp
    gradients[f"dW{L}"] = dW_temp
    gradients[f"db{L}"] = db_temp
    
    # Loop from l=L-2 to l=0 (exclude output & last layers)
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dW_temp, dA_prev_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "relu")
        gradients[f"dA{l}"] = dA_prev_temp
        gradients[f"dW{l+1}"] = dW_temp
        gradients[f"db{l+1}"] = db_temp
    return gradients


#################################################################################################
# Exercise 10
def update_parameters(parameters: dict, gradients: dict, learning_rate: float) -> dict:
    L = len(parameters) // 2 # number of layers in the neural network

    # Perform update
    for l in range(L):
        parameters[f"W{l+1}"] = parameters[f"W{l+1}"] - learning_rate * gradients[f"dW{l+1}"]
        parameters[f"b{l+1}"] = parameters[f"b{l+1}"] - learning_rate * gradients[f"db{l+1}"]
    return parameters