import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets
from opt_utils_v1a import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils_v1a import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from copy import deepcopy
from testCases import *

### Exercise 1 - update_parameters_with_gd
def update_parameters_with_gd(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural networks
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads['dW' + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads['db' + str(l)]
    return parameters
def Exercise_1():
    parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    learning_rate = 0.01
    parameters = update_parameters_with_gd(parameters, grads, learning_rate)
    print("W1 =\n" + str(parameters["W1"]))
    print("b1 =\n" + str(parameters["b1"]))
    print("W2 =\n" + str(parameters["W2"]))
    print("b2 =\n" + str(parameters["b2"]))

### Exercise 2 - random_mini_batches
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)         # tạo ngẫu nhiên minibatches
    m = X.shape[1]               # số lượng training sample
    mini_batches = []

    # Step 1: Xáo trộn(X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    inc = mini_batch_size

    # Step 2 - phân vùng(shuffled_X, shuffled_Y).
    # Các trường hợp chỉ có kích thước lô nhỏ hoàn chỉnh, tức là mỗi trường hợp trong số 64 trường hợp.
    num_complete_minibatches = math.floor( m / mini_batch_size)  #số lô nhỏ có kích thước mini_batch_size trong phân vùng của bạn
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    #Để xử lý trường hợp cuối (lô nhỏ cuối cùng < mini_batch_size tức là nhỏ hơn 64)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, int(m/mini_batch_size)*mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, int(m/mini_batch_size)*mini_batch_size : ]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
np.random.seed(1)
mini_batch_size = 64
nx = 12288
m = 148
X = np.array([x for x in range(nx * m)]).reshape((m, nx)).T
Y = np.random.randn(1, m) < 0.5

mini_batches = random_mini_batches(X, Y, mini_batch_size)
n_batches = len(mini_batches)

n_batches == math.ceil(m / mini_batch_size), f"Wrong number of mini batches. {n_batches} != {math.ceil(m / mini_batch_size)}"
for k in range(n_batches - 1):
    mini_batches[k][0].shape == (nx, mini_batch_size), f"Wrong shape in {k} mini batch for X"
    mini_batches[k][1].shape == (1, mini_batch_size), f"Wrong shape in {k} mini batch for Y"
    np.sum(np.sum(mini_batches[k][0] - mini_batches[k][0][0], axis=0)) == ((nx * (nx - 1) / 2 ) * mini_batch_size), "Wrong values. It happens if the order of X rows(features) changes"
if ( m % mini_batch_size > 0):
    mini_batches[n_batches - 1][0].shape == (nx, m % mini_batch_size), f"Wrong shape in the last minibatch. {mini_batches[n_batches - 1][0].shape} != {(nx, m % mini_batch_size)}"

np.allclose(mini_batches[0][0][0][0:3], [294912,  86016, 454656]), "Wrong values. Check the indexes used to form the mini batches"
np.allclose(mini_batches[-1][0][-1][0:3], [1425407, 1769471, 897023]), "Wrong values. Check the indexes used to form the mini batches"
def Exercise_2():
    t_X, t_Y, mini_batch_size = random_mini_batches_test_case()
    mini_batches = random_mini_batches(t_X, t_Y, mini_batch_size)

    print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
    print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

### Exercise 3 - initialize_velocity
def initialize_velocity(parameters):
    L = len(parameters) // 2 # số lượng lớp ẩn
    v = {}

    # Initialize velocity
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
    return v

def Exercise_3():
    parameters = initialize_velocity_test_case()
    v = initialize_velocity(parameters)
    print("v[\"dW1\"] =\n" + str(v["dW1"]))
    print("v[\"db1\"] =\n" + str(v["db1"]))
    print("v[\"dW2\"] =\n" + str(v["dW2"]))
    print("v[\"db2\"] =\n" + str(v["db2"]))

### Exercise 4 - update_parameters_with_momentum
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta*v["dW" + str(l)] + (1 - beta)*grads['dW' + str(l)]
        v["db" + str(l)] = beta*v["db" + str(l)] + (1 - beta)*grads['db' + str(l)]

        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]

    return parameters, v

def Exercise_4():
    parameters, grads, v = update_parameters_with_momentum_test_case()

    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
    print("W1 = \n" + str(parameters["W1"]))
    print("b1 = \n" + str(parameters["b1"]))
    print("W2 = \n" + str(parameters["W2"]))
    print("b2 = \n" + str(parameters["b2"]))
    print("v[\"dW1\"] = \n" + str(v["dW1"]))
    print("v[\"db1\"] = \n" + str(v["db1"]))
    print("v[\"dW2\"] = \n" + str(v["dW2"]))
    print("v[\"db2\"] = v" + str(v["db2"]))

### Exercise 5 - initialize_adam
def initialize_adam(parameters) :
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
        s["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        s["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
    return v, s

def Exercise_5():
    parameters = initialize_adam_test_case()

    v, s = initialize_adam(parameters)
    print("v[\"dW1\"] = \n" + str(v["dW1"]))
    print("v[\"db1\"] = \n" + str(v["db1"]))
    print("v[\"dW2\"] = \n" + str(v["dW2"]))
    print("v[\"db2\"] = \n" + str(v["db2"]))
    print("s[\"dW1\"] = \n" + str(s["dW1"]))
    print("s[\"db1\"] = \n" + str(s["db1"]))
    print("s[\"dW2\"] = \n" + str(s["dW2"]))
    print("s[\"db2\"] = \n" + str(s["db2"]))


### Exercise 6 - update_parameters_with_adam
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads['db' + str(l)]

        v_corrected["dW" + str(l)] = v["dW" + str(l)]/(1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)]/(1 - beta1**t)

        s["dW" + str(l)] = beta2*s["dW" + str(l)] + (1 - beta2)*np.square(grads['dW' + str(l)])
        s["db" + str(l)] = beta2*s["db" + str(l)] + (1 - beta2)*np.square(grads['db' + str(l)])

        s_corrected["dW" + str(l)] = s["dW" + str(l)]/(1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)]/(1 - beta2**t)

        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*v_corrected["dW" + str(l)]/(np.sqrt(s_corrected["dW" + str(l)])+epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*v_corrected["db" + str(l)]/(np.sqrt(s_corrected["db" + str(l)])+epsilon)

    return parameters, v, s, v_corrected, s_corrected

def Exercise_6():
    parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon = update_parameters_with_adam_test_case()

    parameters, v, s, vc, sc = update_parameters_with_adam(parametersi, grads, vi, si, t, learning_rate, beta1, beta2,
                                                           epsilon)
    print(f"W1 = \n{parameters['W1']}")
    print(f"W2 = \n{parameters['W2']}")
    print(f"b1 = \n{parameters['b1']}")
    print(f"b2 = \n{parameters['b2']}")

### Exercise 7 - update_lr
def update_lr(learning_rate0, epoch_num, decay_rate):
    learning_rate = learning_rate0 / (1 + decay_rate * epoch_num)
    return learning_rate

def Exercise_7():
    learning_rate = 0.5
    print("Original learning rate: ", learning_rate)
    epoch_num = 2
    decay_rate = 1
    learning_rate_2 = update_lr(learning_rate, epoch_num, decay_rate)
    print("Updated learning rate: ", learning_rate_2)



    ###################################################################################
def main() -> None:
    train_X, train_Y = load_dataset()
    # Exercise_1()
    # Exercise_2()
    # Exercise_3()
    # Exercise_4()
    # Exercise_5()
    # Exercise_6()
    Exercise_7()

    return None
if __name__ == '__main__':
    main()

