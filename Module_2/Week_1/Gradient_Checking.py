import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

### Exercise 1 - forward_propagation
def forward_propagation(x, theta):
    # the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
    J = np.dot(theta, x)
    return J
def Exercise_1():
    x, theta = 2, 4
    J = forward_propagation(x, theta)
    print("J = " + str(J))

### Exercise 2 - backward_propagation
def backward_propagation(x, theta):
    # the derivative of J with respect to theta
    dtheta = x
    return dtheta

def Exercise_2():
    x, theta = 3, 4
    dtheta = backward_propagation(x, theta)
    print("dtheta = " + str(dtheta))

### Exercise 3 - gradient_check
def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
    thetaplus = theta + epsilon                     # Step 1
    thetaminus = theta - epsilon                    # Step 2
    J_plus = np.dot(thetaplus,x)                    # Step 3
    J_minus = np.dot(thetaminus,x)                  # Step 4
    gradapprox = (J_plus - J_minus)/(2*epsilon)     # Step 5

    ##Kiểm tra xem gradapprox có đủ gần với đầu ra của back_propagation() không
    grad = x

    numerator = np.linalg.norm(gradapprox-grad)
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)
    difference = numerator/denominator

    if print_msg:
        if difference > 2e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                difference) + "\033[0m")
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
                difference) + "\033[0m")

    return difference
def Exercise_3():
    x, theta = 3, 4
    difference = gradient_check(x, theta, print_msg=True)

## N-Dimensional Gradient Checking
def forward_propagation_n(X, Y, parameters):
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    log_probs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(log_probs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def backward_propagation_n(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) * 2
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

### Exercise 4 - gradient_check_n
def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)

    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))

        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)                    # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator                             # Step 3'

    if print_msg:
        if difference > 2e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                difference) + "\033[0m")
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
                difference) + "\033[0m")

        return difference

def Exercise_4():
    X, Y, parameters = gradient_check_n_test_case()

    cost, cache = forward_propagation_n(X, Y, parameters)
    gradients = backward_propagation_n(X, Y, cache)
    difference = gradient_check_n(parameters, gradients, X, Y, 1e-7, True)
    expected_values = [0.2850931567761623, 1.1890913024229996e-07]
    type(difference) == np.ndarray
    np.any(np.isclose(difference, expected_values))

##########################################################
def main() -> None:
    # Exercise_1()
    # Exercise_2()
    # Exercise_3()
    Exercise_4()

    return None
if __name__ == '__main__':
    main()