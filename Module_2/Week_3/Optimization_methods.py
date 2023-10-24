import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils_v1a import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils_v1a import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from copy import deepcopy




###############################################################################
def update_parameters_with_gd(parameters: dict, grads: dict, learning_rate: float) -> dict:
    """
    Arguments:
    parameters (dict):
        Ex: parameters['W' + str(l)] = Wl

    grads (dict):
        Ex:grads['dW' + str(l)] = dWl
    Output (dict): containing your updated parameters 
    """

    # number of layers in the neural networks
    L = len(parameters) // 2

    # Update rule for each parameter
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters


###############################################################################
def random_mini_batches(X: np.ndarray, Y: np.ndarray, mini_batch_size = 64, seed = 0) -> list:
    """
    X: (input size, number of examples)
    Y: (1, number of examples)
    
    Returns:
    mini_batches: list of random sampling mini batch
    """
    
    np.random.seed(seed)
    num_of_examples = X.shape[1]
    mini_batches = []
        
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(num_of_examples))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, num_of_examples))

    # Partition (shuffled_X, shuffled_Y).
    num_complete_minibatches = num_of_examples // mini_batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k : mini_batch_size * (k+1)]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # handling the end case (last mini-batch < mini_batch_size)
    if num_of_examples % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * (k+1):]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * (k+1):]        
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


################################################################################
def initialize_velocity(parameters):
    # Output: v (dict): containing the current velocity

    # number of layers in the neural networks
    L = len(parameters) // 2
    velocity = {}
    
    # Init velocity
    for l in range(1, L + 1):
        velocity[f"dW{l}"] = np.zeros(parameters[f"W{l}"].shape)
        velocity[f"db{l}"] = np.zeros(parameters[f"b{l}"].shape)
    return velocity


def update_parameters_with_momentum(parameters: dict, grads: dict, velocity: dict, beta: float, learning_rate: float) -> tuple:
    # number of layers in the neural networks
    L = len(parameters) // 2 
    
    # Update momentum
    for l in range(1, L + 1):
        velocity[f"dW{l}"] = beta * velocity[f"dW{l}"] +\
                            (1-beta) * grads[f"dW{l}"]

        velocity[f"db{l}"] = beta * velocity[f"db{l}"] +\
                            (1-beta) * grads[f"db{l}"]

        parameters[f"W{l}"] -= learning_rate * v[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * v[f"db{l}"]        
    return parameters, v


##############################################################################
def initialize_adam(parameters):
    # number of layers in the neural networks
    L = len(parameters) // 2
    velocity = {}
    s_w = {}
    
    for l in range(1, L + 1):
        # Init velocity
        velocity[f"dW{l}"] = np.zeros(parameters[f"W{l}"].shape)
        velocity[f"db{l}"] = np.zeros(parameters[f"b{l}"].shape)

        # Init s_w
        s_w[f"dW{l}"] = np.zeros(parameters[f"W{l}"].shape)
        s_w[f"db{l}"] = np.zeros(parameters[f"b{l}"].shape)
    return velocity, s_w


def update_parameters_with_adam(parameters: dict, grads: dict,
    velocity: dict, s_w: dict,
    taken_steps: int, learning_rate: float = 0.01,
    beta1: float = 0.9, beta2: float = 0.999, 
    epsilon: float = 1e-8):
    """
    velocity (dict): EWA of 1st grad
        -> beta_1
    
    s_w (dict): EWA of squared gradient
        -> beta_2
    
    taken_steps: counts the number of taken steps

    epsilon -- circumvent zero division
    """
    # number of layers in the neural 
    L = len(parameters) // 2

    velocity_corrected = {}
    s_w_corrected = {}
    
    # Update Adam optimizer
    for l in range(1, L + 1):
        # calculate EWA
        velocity[f"dW{l}"] = beta1 * velocity[f"dW{l}"] +\
                            (1-beta1) * grads[f"dW{l}"]

        velocity[f"db{l}"] = beta1 * velocity[f"db{l}"] +\
                            (1-beta1) * grads[f"db{l}"]
        # bias correction
        velocity_corrected[f"dW{l}"] = v[f"dW{l}"] / (1-np.power(beta1, t))
        velocity_corrected[f"db{l}"] = v[f"db{l}"] / (1-np.power(beta1, t))
        
        # calculate update_parameters_with_adam
        s_w[f"dW{l}"] = beta2 * s_w[f"dW{l}"] +\
                       (1 - beta2) * np.square(grads[f"dW{l}"])

        s_w[f"db{l}"] = beta2 * s_w[f"db{l}"] +\
                       (1 - beta2) * np.square(grads[f"db{l}"])
        
        # bias correction
        s_w_corrected[f"dW{l}"] = s_w[f"dW{l}"] / (1 - beta2 ** taken_steps)
        s_w_corrected[f"db{l}"] = s_w[f"db{l}"] / (1 - beta2 ** taken_steps)

        # Update paras 
        parameters[f"W{l}"] -= learning_rate * (v_corrected[f"dW{l}"] / (np.sqrt(s_w_corrected[f"dW{l}"]) + epsilon))
        parameters[f"b{l}"] -= learning_rate * (v_corrected[f"db{l}"] / (np.sqrt(s_w_corrected[f"db{l}"]) + epsilon))
    return parameters, velocity, s_w, velocity_corrected, s_w_corrected


###############################################################################
def model(X: np.ndarray, Y: np.ndarray, layers_dims: int, 
          optimizer: str, learning_rate = 0.0007, mini_batch_size = 64,
          beta = 0.9, beta1 = 0.9, beta2 = 0.999, 
          epsilon = 1e-8, num_epochs = 5000, print_cost = True):
    L = len(layers_dims)             
    costs = []                       
    taken_steps = 0
    seed = 10
    num_of_examples = X.shape[1]
    parameters = initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        velocity = initialize_velocity(parameters)
    elif optimizer == "adam":
        velocity, s_w = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_epochs):
        # increment the seed to reshuffle after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            # Forward prop
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost
            cost_total += compute_cost(a3, minibatch_Y)

            # Backprop
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update paras
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                taken_steps = taken_steps + 1
                parameters, velocity, s_w, _, _ = update_parameters_with_adam(parameters, grads,
                                                                     velocity, s_w, taken_steps,
                                                                     learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / num_of_examples
        
        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return parameters


###############################################################################
def Mini_Batch_GD(train_X, train_Y) -> None:
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    return None


def Mini_Batch_GD_with_Momentum(train_X, train_Y) -> None:
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Momentum optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    return None


def Mini_Batch_with_Adam(train_X, train_Y) -> None:
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    return None


#######################################################################################################
# Learning Rate Decay and Scheduling -> Speed up feed forward
# Do later

#######################################################################################################
def main() -> None:
    train_X, train_Y = load_dataset() # (2, 300) & (1, 300)
    Mini_Batch_GD(train_X, train_Y)
    Mini_Batch_GD_with_Momentum(train_X, train_Y)
    Mini_Batch_with_Adam(train_X, train_Y)
    return None


if __name__ == '__main__':
    main()