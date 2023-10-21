import numpy as np

from pprint import pprint as pp
from rnn_utils import softmax, sigmoid, initialize_adam, update_parameters_with_adam


def rnn_cell_forward(xt, a_prev, parameters):
    """
    Inp
    xt: inp at timestep "t"
        -> (n_x, m)
    a_prev: hidden state at timestep "t-1"
        -> (n_a, m)
    parameters: dict contains
        Wax - (n_a, n_x)
        Waa - (n_a, n_a)
        Wya - (n_y, n_a)
        ba - (n_a, 1)
        by - (n_y, 1)

    Returns:
        a_next - (n_a, m)
        yt_pred - (n_y, m)
    cache - (a_next, a_prev, xt, parameters)
    """
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache


def Exercise_1() -> None:
    np.random.seed(1)

    xt = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)

    parameters = {}
    parameters['Wax'] = np.random.randn(5, 3)
    parameters['Waa'] = np.random.randn(5, 5)
    parameters['Wya'] = np.random.randn(2, 5)
    parameters['ba'] = np.random.randn(5, 1)
    parameters['by'] = np.random.randn(2, 1)

    a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
    print("a_next.shape = \n", a_next.shape)
    print("yt_pred.shape = \n", yt_pred.shape)
    return None


############################################################################################################################
def rnn_forward(x, a0, parameters):
    """
    Inp
        x: inputs
            -> (n_x, m, T_x)
        a0: Initial hidden state
            -> (n_a, m)
        parameters
            Waa - (n_a, n_a)
            Wax - (n_a, n_x)
            Wya - (n_y, n_a)
            ba - (n_a, 1)
            by - (n_y, 1)

    Returns:
    a: Hidden states for every time-step
        -> (n_a, m, T_x)
    y_pred: Predictions for every time-step
        -> (n_y, m, T_x)
    caches: list of cached paras
    """
    caches = []

    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape


    # initialize "a" and "y_pred"
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize a_next
    a_next = a0

    # loop over all time-steps
    for t in range(T_x):
        # forward pass
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)

        # save hidden_state & y_pred for compute lost at time step t
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred

        # cache paras
        caches.append(cache)
    caches = (caches, x)
    return a, y_pred, caches


def Exercise_2() -> None:
    np.random.seed(1)
    input_size = 3
    hidden_state_size = 5
    mini_batch_size = 10
    time_step = 4

    x = np.random.randn(input_size, mini_batch_size, time_step)
    a0 = np.random.randn(hidden_state_size, mini_batch_size)

    parameters = {}
    parameters['Waa'] = np.random.randn(5, 5)
    parameters['Wax'] = np.random.randn(5, 3)
    parameters['Wya'] = np.random.randn(2, 5)
    parameters['ba'] = np.random.randn(5, 1)
    parameters['by'] = np.random.randn(2, 1)

    a, y_pred, caches = rnn_forward(x, a0, parameters)

    print("a.shape =", a.shape)
    print("y_pred.shape =", y_pred.shape)
    return None


#############################################################################################################################



def main() -> None:
    # RNN
    # Exercise_1()
    Exercise_2()

    # LSTM
    
    return None

if __name__ == '__main__':
    main()