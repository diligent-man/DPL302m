import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model


########################################################################################################
# Provided functions
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(nets.make_circl_samples==N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure


################################################################################
def Exercise_1(X: np.ndarray, Y: np.ndarray) -> None:
    shape_X = X.shape
    shape_Y = Y.shape
    m = shape_X[1]  # training set size

    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print ('I have m = %d training examples!' % (m))
    return None


################################################################################
def layer_sizes(X, Y):
    """
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    inp_size = X.shape[0]
    hidden_size = 4
    output_size = Y.shape[0]
    return (inp_size, hidden_size, output_size)


def Exercise_2(X: np.ndarray, Y:np.ndarray) -> tuple:
    (inp_size, hidden_size, output_size) = layer_sizes(X, Y)
    # print("The size of the input layer is: " + str(inp_size))
    # print("The size of the hidden layer is: " + str(hidden_size))
    # print("The size of the output layer is: " + str(output_size))
    return inp_size, hidden_size, output_size


################################################################################
def initialize_parameters(inp_size: int, hidden_size: int, output_size: int) -> dict:
    # set up a seed so that your output matches ours although the initialization is random.
    # Anyway, we can init with zero
    np.random.seed(2) 
    
    # Parameters init
    W1 = np.random.randn(hidden_size, inp_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    
    assert (W1.shape == (hidden_size, inp_size))
    assert (b1.shape == (hidden_size, 1))
    assert (W2.shape == (output_size, hidden_size))
    assert (b2.shape == (output_size, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def Exercise_3(inp_size: int, hidden_size: int, output_size: int) -> dict:
    return initialize_parameters(inp_size, hidden_size, output_size)


################################################################################
def forward_prop(X, parameters):
    """
    Returns:
    A2: The sigmoid output of the second activation
    cache: a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    # Retrieve each parameter
    W1 = parameters["W1"] # 4 x 2
    b1 = parameters["b1"] # 4 x 1
    W2 = parameters["W2"] # 1 x 4
    b2 = parameters["b2"] # 1 x 1

    # Forward Prop
    # Inp -> hidden layer
    Z1 = np.add(W1 @ X, b1)
    A1 = np.tanh(Z1)
    # Hidden layer -> Output layer
    Z2 = np.add(W2 @ A1, b2)
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    # cach for back_prop calculation
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


def Exercise_4(X, parameters) -> tuple:
    return forward_propagation(X, parameters)


################################################################################
def compute_cost(A2, Y) -> np.ndarray:
    m = Y.shape[1] # number of examples
    cost = (-1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    cost = float(np.squeeze(cost))
    return cost


def Exercise_5(A2, Y) -> np.ndarray:
    return compute_cost(A2, Y)


###############################################################################
def backward_prop(parameters: dict, cache: dict, X: np.ndarray, Y: np.ndarray) -> dict:
    m = Y.shape[1]

    # First, retrieve W1 and W2
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve A1, A2
    A1 = cache['A1']
    A2 = cache['A2']
    
    # Backward prop: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dZ1 = (W2.T @ dZ2) * (1 - A1**2)
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
    return gradients


def Exercise_6(parameters: dict, cache: dict, X: np.ndarray, Y: np.ndarray) -> tuple:
    return backward_prop(parameters, cache, X, Y)


################################################################################
def update_parameters(parameters: dict, gradients: dict, learning_rate: float) -> dict:
    W1 = parameters["W1"] - (learning_rate * gradients["dW1"]) 
    b1 = parameters["b1"] - (learning_rate * gradients["db1"]) 
    W2 = parameters["W2"] - (learning_rate * gradients["dW2"]) 
    b2 = parameters["b2"] - (learning_rate * gradients["db2"]) 
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def Exercise_7(parameters: dict, gradients: dict, learning_rate: float) -> dict:
    return update_parameters(parameters, gradients, learning_rate)

################################################################################
def nn_model(X: np.ndarray, Y: np.ndarray, hidden_size: int, learning_rate: float=0.05, num_iterations: int = 10000, print_cost: bool=False) -> dict:    
    np.random.seed(3)
    inp_size = layer_sizes(X, Y)[0]
    output_size = layer_sizes(X, Y)[2]

    # Init
    parameters = initialize_parameters(inp_size, hidden_size, output_size)
    
    # gradient descent
    for i in range(0, num_iterations):
        A2, cache = forward_prop(X, parameters)
        cost = compute_cost(A2, Y)
        gradients = backward_prop(parameters, cache, X, Y)
        parameters = update_parameters(parameters, gradients, learning_rate=learning_rate)
        
        # Print the cost every 10 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters


def Exercise_8(X: np.ndarray, Y: np.ndarray, hidden_size: int, learning_rate: float=0.05, num_iterations: int=10000, print_cost: bool=False) -> dict:
    return nn_model(X, Y, hidden_size, learning_rate, num_iterations, print_cost)


################################################################################
def predict(parameters: dict, X: np.ndarray) -> np.ndarray:
    A2, cache = forward_prop(X, parameters)
    y_predict = np.where(A2 > 0.5, 1, 0)
    return y_predict


def Exercise_9(parameters: dict, X: np.ndarray) -> np.ndarray:
    return predict(parameters, X)


################################################################################
def simple_logistic_regression(X: np.ndarray, Y: np.ndarray) -> None:
    model = sklearn.linear_model.LogisticRegressionCV()
    model.fit(X.T, Y.T)

    # Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: model.predict(x), X, Y)
    plt.title("Logistic Regression")
    # plt.show()

    # Print accuracy
    LR_predictions = model.predict(X.T)
    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
           '% ' + "(percentage of correctly labelled datapoints)")
    # Remark: This dataset is not linearly separable,
    #         so logistic regression doesn't perform well.
    #         Hopefully a neural network will do better. Let's try this now!
    return None


def main() -> None:
    # X: 2 features
    # Y: 2 labels (red:0, blue:1)
    X, Y = load_planar_dataset() # row vecs
    
    # Visualize the data:
    # plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    # plt.show()

    # Exercise_1(X, Y)
    # simple_logistic_regression(X, Y)
    
    # layer_sizes
    inp_size, hidden_size, output_size =  Exercise_2(X, Y) 
    
    # initialize_parameters
    parameters = Exercise_3(inp_size, hidden_size, output_size) 
    
    # forward_propagation
    # A2, cache = Exercise_4(X, parameters)
    
    # cost_func
    # cost = Exercise_5(A2, Y)
    
    # backward_propagation
    # gradients = Exercise_6(parameters, cache, X, Y)
    
    # gradient descent
    # parameters = Exercise_7(parameters, gradients)
    
    # Incorporate into nn_model
    # parameters = Exercise_8(X, Y, hidden_size=100, learning_rate=0.5, num_iterations=10000, print_cost=True)
    # y_predict = Exercise_9(parameters, X)
    
    # Plot the decision boundary
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # plt.title("Decision Boundary for hidden layer size " + str(6))
    # plt.show()

    # Print accuracy
    # cross_entropy_part_1 = Y @ y_predict.T
    # cross_entropy_part_2 = (1 - Y) @ (1 - y_predict.T)
    # print(f'Accuracy: {np.squeeze(cross_entropy_part_1 + cross_entropy_part_2) / Y.size:.1}')
    return None


if __name__ == '__main__':
    main()

