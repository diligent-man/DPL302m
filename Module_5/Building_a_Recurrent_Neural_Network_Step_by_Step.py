import numpy as np

from numpy.random import randn
from pprint import pprint as pp
from rnn_utils import *


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
    Waa = parameters["Waa"]
    Wax = parameters["Wax"]; ba = parameters["ba"]
    Wya = parameters["Wya"]; by = parameters["by"]

    a_next = np.tanh(Waa @ a_prev + Wax @ xt + ba)
    yt_pred = softmax(Wya @ a_next + by)

    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache


def Exercise_1() -> None:
    np.random.seed(1)

    xt = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)

    paras = {}
    paras['Waa'] = randn(5, 5)
    paras['Wax'] = randn(5, 3); paras['ba'] = randn(5, 1)
    paras['Wya'] = randn(2, 5); paras['by'] = randn(2, 1)
        
    a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, paras)
    print("a_next.shape = \n", a_next.shape)
    print("yt_pred.shape = \n", yt_pred.shape)
    return None


############################################################################################################################
def rnn_forward(x, a0, paras):
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
    n_y, n_a = paras["Wya"].shape

    # init "a" and "y_pred"
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize a_next
    a_next = a0

    # loop over all time-steps
    for t in range(T_x):
        # forward pass
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, paras)

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

    x = randn(input_size, mini_batch_size, time_step)
    a0 = randn(hidden_state_size, mini_batch_size)

    paras = {}
    paras['Waa'] = randn(hidden_state_size, hidden_state_size)
    
    paras['Wax'] = randn(hidden_state_size, input_size)
    paras['ba'] = randn(hidden_state_size, 1)
    
    paras['Wya'] = randn(2, hidden_state_size)
    paras['by'] = randn(hidden_state_size, 1)
    
    a, y_pred, caches = rnn_forward(x, a0, paras)
    print("a.shape =", a.shape)
    print("y_pred.shape =", y_pred.shape)
    return None


#############################################################################################################################
def lstm_cell_forward(xt, a_prev, c_prev, paras):
    """
    Inp:
        xt: inp at t
            -> (n_x, m).
        a_prev: hidden state at t-1
            -> (n_a, m)
        c_prev: (Long-term) mem at t-1"
            -> (n_a, m)
        parameters: contains Wf, Wi, Wc, Wo, Wy are respectively weights for
            forget gate, update gate, candidate mem, output gate, prediction
            -> (n_a, n_a + n_x)
            
            bf, bi, bc, bo, by are corresponding biases
            -> (n_a, 1) | (n_y, 1)
                        
    Out:
        a_next: next hidden state
        -> (n_a, m)
        
        c_next: next long-term mem
        -> (n_a, m)
        
        yt_pred: prediction at timestep t
        -> (n_y, m)
        
        cache: contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    """

    # Retrieve parameters from "parameters"
    Wf = paras["Wf"]; bf = paras["bf"]
    Wi = paras["Wi"]; bi = paras["bi"]
    Wc = paras["Wc"]; bc = paras["bc"]
    Wo = paras["Wo"]; bo = paras["bo"]
    Wy = paras["Wy"]; by = paras["by"]
    
    # Retrieve shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.concatenate([a_prev, xt], axis=0)

    # compute necessary values
    forget_gate = sigmoid(Wf @ concat + bf)
    inp_gate = sigmoid(Wi @ concat + bi)
    candidate_mem = np.tanh(Wc @ concat + bc)
    out_gate = sigmoid(Wo @ concat + bo)

    c_next = forget_gate * c_prev + inp_gate * candidate_mem
    a_next = out_gate * np.tanh(c_next)
    yt_pred = softmax(Wy @ a_next + by)

    # cach vals
    cache = (a_next, c_next, a_prev, c_prev,
             forget_gate, inp_gate, candidate_mem, out_gate, xt, paras)
    return a_next, c_next, yt_pred, cache


def Exercise_3() -> None:
    np.random.seed(1)
    
    input_size = 3
    hidden_state_size = 5
    mini_batch_size = 10
    output_size = 2
    xt = randn(input_size, mini_batch_size)
    a_prev = randn(hidden_state_size, mini_batch_size)
    c_prev = randn(hidden_state_size, mini_batch_size)
    
    paras = {}
    paras['Wf'] = randn(hidden_state_size, hidden_state_size + input_size)
    paras['bf'] = randn(hidden_state_size, 1)
    
    paras['Wi'] = randn(hidden_state_size, hidden_state_size + input_size)
    paras['bi'] = randn(hidden_state_size, 1)
    
    paras['Wo'] = randn(hidden_state_size, hidden_state_size + input_size)
    paras['bo'] = randn(hidden_state_size, 1)
    
    paras['Wc'] = randn(hidden_state_size, hidden_state_size + input_size)
    paras['bc'] = randn(hidden_state_size, 1)
    
    paras['Wy'] = randn(output_size, hidden_state_size)
    paras['by'] = randn(output_size, 1)

    a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, paras)

    print("a_next =", a_next.shape)
    print("c_next =", c_next.shape)
    print("yt =", yt.shape)
    print("len(cache) =", len(cache))
    return None


##########################################################################
def lstm_forward(x, a0, paras):
    """
    Inp:
        x: inp
        a0: Initial hidden state
        paras: contains Wf, Wi, Wc, Wo, Wy are respectively weights for
            forget gate, update gate, candidate mem, output gate, prediction
            -> (n_a, n_a + n_x)
            
            bf, bi, bc, bo, by are corresponding biases
            -> (n_a, 1) | (n_y, 1)
                        
    Out:
        a: hidden states for every time-step
            -> (n_a, m, T_x)
        y: predictions for every time-step,
            -> (n_y, m, T_x)
        c: long-term mem
            -> (n_a, m, T_x)
        caches: (list of all the caches, x)
    """

    # Init caches
    caches = []
    
    # Retrieve inp
    Wy = paras['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    # initi "a", "c" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    c = np.zeros(a.shape)
    y = np.zeros((n_y, m, T_x))
    
    # Initialize a_next and c_next
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    # loop over all time-steps
    for t in range(T_x):
        xt = x[:, :, t]
        # forward pass
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, paras)

        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt

        caches.append(cache)
    
    # store values needed for backprop in cache
    caches = (caches, x)
    return a, y, c, caches


def Exercise_4() -> None:
    np.random.seed(123)
    
    input_size = 300
    hidden_state_size = 500
    mini_batch_size = 100
    output_size = 200
    T = 100

    # init
    x = randn(input_size, mini_batch_size, T)
    a0 = randn(hidden_state_size, mini_batch_size)

    paras = {}
    paras['Wf'] = randn(hidden_state_size, hidden_state_size + input_size)
    paras['bf'] = randn(hidden_state_size, 1)
    
    paras['Wi'] = randn(hidden_state_size, hidden_state_size + input_size)
    paras['bi'] = randn(hidden_state_size, 1)
    
    paras['Wo'] = randn(hidden_state_size, hidden_state_size + input_size)
    paras['bo'] = randn(hidden_state_size, 1)
    
    paras['Wc'] = randn(hidden_state_size, hidden_state_size + input_size)
    paras['bc'] = randn(hidden_state_size, 1)
    
    paras['Wy'] = randn(output_size, hidden_state_size)
    paras['by'] = randn(output_size, 1)

    a, c, yt, caches = lstm_forward(x, a0, paras)
    print("a.shape = ", a.shape)
    print("y.shape = ", yt.shape)
    print("len(caches) = ", len(caches))
    return None


###############################################################################
def rnn_cell_backward(da_next, cache):
    """
    Inp
    da_next: Gradient of loss with respect to next hidden state
    cache

    Out:
        gradients: contains
            da_prev: Grads of prev hidden state (n_a, m)
            
            dx: Grads of input data (n_x, m)
            dWax: Grads of input weight (n_a, n_x)

            dWaa: Grads of hidden state weight (n_a, n_a)
            dba: Gradients of bias (n_a, 1)
"""
    # Retrieve values from cache
    (a_next, a_prev, xt, paras) = cache
    
    # Retrieve values from parameters
    Waa = paras["Waa"]
    Wax = paras["Wax"]; ba = paras["ba"]
    Wya = paras["Wya"]; by = paras["by"]
    
    # compute the gradient of dtanh
    dtanh = da_next * (1 - np.square(a_next))

    # compute the gradient of the loss with respect to Wax
    dxt = Wax.T @ dtanh
    dWax = dtanh @ xt.T

    # compute the gradient with respect to Waa
    da_prev = Waa.T @ dtanh
    dWaa = dtanh @ a_prev.T

    # compute the gradient with respect to b
    dba = np.sum(dtanh, axis=-1, keepdims=True)

    # Store the gradients in a python dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients


def Exercise_5() -> None:
    np.random.seed(1)
    
    xt = randn(3, 10)
    a_prev = randn(5, 10)
    
    paras = {}
    paras['Waa'] = randn(5, 5)
    paras['Wax'] = randn(5, 3); paras['ba'] = randn(5, 1)
    paras['Wya'] = randn(2, 5); paras['by'] = randn(2, 1)
    
    a_next, yt, cache = rnn_cell_forward(xt, a_prev, paras)

    da_next = randn(5, 10)
    gradients = rnn_cell_backward(da_next, cache)
    
    print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
    print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
    print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
    print("gradients[\"dba\"].shape =", gradients["dba"].shape)
    return None


###############################################################################
def rnn_backward(da, caches):
    """
    Inp
        da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
        caches -- tuple containing information from the forward pass (rnn_forward)
    
    Out
        gradients: contains
        dx: Grad w.r.t. inp (n_x, m, T_x)
        da0: Grad w.r.t initial hidden state (n_a, m)
        dWax: Grad w.r.t input's weight matrix (n_a, n_x)
        dWaa: Grad w.r.t hidden state's weight matrix (n_a, n_a)
        dba: Gradient w.r.t the bias (n_a, 1)
    """
    # Retrieve values from the first cache
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    
    # Retrieve dim from da & x1
    n_a, m, T_x = da.shape
    n_x, m = x1.shape 
    
    # init the grads
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x)); dba = np.zeros((n_a, 1))
    dWaa = np.zeros((n_a, n_a))
    
    da0 = np.zeros((n_a, m))
    da_prev_t = np.zeros(da0.shape)
    
    # Loop from last t
    for t in reversed(range(T_x)):
        # Compute gradients at time step t
        gradients = rnn_cell_backward(da[:, :, t] + da_prev_t, caches[t])
        
        # Retrieve derivatives from gradients (≈ 1 line)
        dxt, da_prev_t, dWax_t, dWaa_t, dba_t = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        
        # Increment global derivatives w.r.t parameters
        dx[:, :, t] = dxt
        dWax += dWax_t
        dWaa += dWaa_t
        dba += dba_t
        
    # Set da0 to the grad of which has been backpropagated through all time
    da0 = da_prev_t

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients


def Exercise_6() -> None:
    np.random.seed(1)

    input_size = 3
    hidden_state_size = 5
    mini_batch_size = 10
    output_size = 2
    time_step = 4

    xt = randn(input_size, mini_batch_size)
    a_prev = randn(hidden_state_size, mini_batch_size)
    c_prev = randn(hidden_state_size, mini_batch_size)

    x = randn(input_size, mini_batch_size, time_step)
    a0 = randn(hidden_state_size, mini_batch_size)

    paras = {}
    paras['Wax'] = randn(hidden_state_size, input_size)
    paras['Waa'] = randn(hidden_state_size, hidden_state_size)
    paras['Wya'] = randn(output_size, hidden_state_size)
    paras['ba'] = randn(hidden_state_size, 1)
    paras['by'] = randn(output_size, 1)

    a, y, caches = rnn_forward(x, a0, paras)
    da_tmp = randn(hidden_state_size, mini_batch_size, time_step)
    gradients_tmp = rnn_backward(da_tmp, caches)
    
    print("gradients[\"dx\"].shape =", gradients_tmp["dx"].shape)
    print("gradients[\"da0\"].shape =", gradients_tmp["da0"].shape)
    print("gradients[\"dWax\"].shape =", gradients_tmp["dWax"].shape)
    print("gradients[\"dWaa\"].shape =", gradients_tmp["dWaa"].shape)
    print("gradients[\"dba\"].shape =", gradients_tmp["dba"].shape)
    return None


###############################################################################
def main() -> None:
    # RNN
    # Exercise_1()
    # Exercise_2()

    # LSTM
    # Exercise_3()
    # Exercise_4()
    # Exercise_5()

    # Backprop in RNN
    # Exercise_6()
    return None


if __name__ == '__main__':
    main()