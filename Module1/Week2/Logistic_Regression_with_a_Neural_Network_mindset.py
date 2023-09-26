import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
    # More details in hdf5: https://docs.h5py.org/en/stable/high/group.html
    # h5py.File works as a dict
    with h5py.File('train_catvnoncat.h5', "r") as train_set:
        train_x = np.array(train_set["train_set_x"][:]) # dim = (209,64,64,3)
        train_y = np.array(train_set["train_set_y"][:]) # dim = (209)
    train_y = train_y.reshape(1, -1) # dim = (209,1)

    with h5py.File('test_catvnoncat.h5', "r") as test_set:
        test_x = np.array(test_set["test_set_x"][:]) # dim = (209,64,64,3)
        test_y = np.array(test_set["test_set_y"][:]) # dim = (209)
        classes = np.array(test_set["list_classes"][:]) # dim =(2)
    test_y = test_y.reshape((1, test_y.shape[0])) # dim = (209,1)
    classes = np.char.decode(classes, encoding="utf-8")
    return train_x, train_y, test_x, test_y, classes


def check_img(i: int, train_x: list, train_y: list, classes: str) -> None:
    print ("y = " + str(train_y[:, i])[1] + ". Thus, it's a " + classes[np.squeeze(train_y[:, i])] + " picture.")

    plt.imshow(train_x[i])
    plt.show()
    return None
###############################################################################


def Exercise_1(train_x: list, train_y: list, test_x: list, test_y: list) -> None:
    m_train = train_x.shape[0]
    m_test =  test_x.shape[0]
    num_px = train_x.shape[1]


    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_x.shape))
    print ("train_set_y shape: " + str(train_y.shape))
    print ("test_set_x shape: " + str(test_x.shape))
    print ("test_set_y shape: " + str(test_y.shape))
    return None


#################################################################################################
def Exercise_2(train_x: list, train_y: list, test_x: list, test_y: list) -> tuple:
    # each img is a col vec
    train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
    test_x_flatten = test_x.reshape(test_x.shape[0], -1).T
    

    assert np.alltrue(train_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
    assert np.alltrue(test_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."

    # print ("train_set_x_flatten shape: " + str(train_x_flatten.shape))
    # print ("train_set_y shape: " + str(train_y.shape))
    # print ("test_set_x_flatten shape: " + str(test_x_flatten.shape))
    # print ("test_set_y shape: " + str(test_y.shape))

    # Standardize/ Normalize
    # Note: with img, don't need to subtract mean
    train_x = np.divide(train_x_flatten, 255)
    test_x = np.divide(test_x_flatten, 255)
    return train_x, test_x


################################################################################
def sigmoid(z) -> list:
    s = 1 / (1 + np.exp(-z))
    return s


def Exercise_3() -> None:
    print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
    return None


################################################################################
def initialize_with_zeros(dim) -> list:
    w = np.zeros((dim, 1))
    b = 0.
    return w, b


def Exercise_4() -> None:
    dim = 2
    w, b = initialize_with_zeros(dim)

    assert type(b) == float
    # print ("w =\n" + str(w))
    # print ("b =\n" + str(b))
    return None


################################################################################
def propagation(W: list, b: float, X: list, Y: list) -> tuple:
    m = X.shape[1] # num of col vecs
    
    # forward propagation: sigmoid(w.T* )
    # A: activation
    A = sigmoid(W.T @ X + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    # backward propagation
    dW = (1/m) * (X @ (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    gradients = {"dW": dW,
                 "db": db}
    return gradients, cost


def Exercise_5() -> None:
    W =  np.array([[1.], [2]])
    b = 1.5
    X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
    Y = np.array([[1, 1, 0]])
    gradients, cost = propagation(W, b, X, Y)

    assert type(gradients["dW"]) == np.ndarray
    assert gradients["dW"].shape == (2, 1)
    assert type(gradients["db"]) == np.float64

    print ("dw = " + str(gradients["dW"]))
    print ("db = " + str(gradients["db"]))
    print ("cost = " + str(cost))
    return None


################################################################################
def optimize(W: list, b: float, X: list, Y: list,
             num_iterations: int, learning_rate: float,
             print_cost=False) -> tuple:
    costs = []

    for i in range(num_iterations):        
        gradients, cost = propagation(W, b, X, Y)

        # Update parameters
        W = W - learning_rate * gradients["dW"]
        b = b - learning_rate * gradients["db"]

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Record & Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after 100 iteration %i: %f" %(i, cost))
    
    parameters = {"W": W,
                  "b": b}
    
    gradients = {"dW": gradients["dW"],
                 "db": gradients["db"]}
    return parameters, gradients, costs


def Exercise_6() -> None:
    # Para init
    W =  np.array([[1.], [2]])
    b = 1.5
    X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
    Y = np.array([[1, 1, 0]])

    parameters, gradients, costs = optimize(W, b, X, Y,
                                    num_iterations=100, learning_rate=0.009,
                                    print_cost=False)

    print ("W = " + str(parameters["W"])); print()
    print ("b = " + str(parameters["b"])); print()
    print ("dW = " + str(gradients["dW"])); print()
    print ("db = " + str(gradients["db"]))
    print("Costs = " + str(costs))
    return None


###############################################################################
def predict(W, b, X) -> list:
    # Use trained/ learned W, b for prediction
    m = X.shape[1]
    threshold = .5
    W = W.reshape(X.shape[0], 1) # row vec

    # step 1: calculate y_predict = A = 𝜎(W.T @ 𝑋 + 𝑏)
    A = sigmoid(W.T @ X + b)

    # step 2: classify
    cat = 1
    non_cat = 0

    # Way 1: vectorization -> faster
    y_predict = np.where(A > threshold, cat, non_cat)

    # Way 2: loop -> slower
    # y_predict = np.zeros_like(A)
    # for i in range(A.shape[1]):
    #     y_predict[0][i] = non_cat if (A[0][i]<=0.5) else cat
    return y_predict


def Exercise_7() -> None:
    # Use trained results for computing prediction
    W = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1., -1.1, -3.2],[1.2, 2., 0.1]])
    print("predictions = " + str(predict(W, b, X)))
    return None


##############################################################################
def Exercise_8(X_train: np.ndarray, Y_train: np.ndarray,
               X_test: np.ndarray, Y_test: np.ndarray,
               classes: list, num_iterations: int = 2000,
               learning_rate: float = 0.005,
               print_cost: bool = False) -> dict:
    W, b = initialize_with_zeros(X_train.shape[0])

    parameters, gradients, costs = optimize(W, b, X_train, Y_train,
                                            num_iterations, learning_rate,
                                            print_cost = print_cost)
    
    # Retrieve parameters W and b from dictionary "parameters"
    W = parameters["W"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_predict_test = predict(W, b, X_test)
    Y_predict_train = predict(W, b, X_train)

    # Print train/test errs - MAE
    MAE = np.mean(np.abs(Y_predict_train - Y_train)) * 100
    print(f"train err: {MAE:.3} %")

    MAE = np.mean(np.abs(Y_predict_test - Y_test)) * 100
    print(f"test err: {MAE:.3} %")

    result = {"costs": costs,
         "y_predict_test": Y_predict_test, 
         "y_predict_train" : Y_predict_train, 
         "W" : W, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations
        }
    return result


###############################################################################
def learning_curve_plotting(logistic_regression_model: dict) -> None:
    # Plot the cost function and the gradients.
    # Plot learning curve (with costs)
    costs = np.squeeze(logistic_regression_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = " + str(logistic_regression_model["learning_rate"]))
    plt.show()
    return None


def multiple_learning_testing(train_x: np.ndarray, train_y: np.ndarray,
                              test_x: np.ndarray, test_y: np.ndarray,
                              classes: list, num_iterations, print_cost: bool) -> None:
    learning_rate_lst = [0.001, 0.001, 0.01, 0.1, 0.5, 1]
    costs_lst = []
    models = {}

    for lr in learning_rate_lst:
        print ("Training a model with learning rate: " + str(lr))
        models[str(lr)] = Exercise_8(train_x, train_y, test_x, test_y, classes, num_iterations=20000, learning_rate=lr, print_cost=print_cost)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for lr in learning_rate_lst:
        plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Multiple Learning rate")
    
    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    return None


def main() -> None:
    # Problem Statement: build a simple image-recognition algo
    # that can correctly classify pictures as cat or non-cat.

    # Dataset info:
    #   + test & training set of m_train images labeled as cat (y=1) or non-cat (y=0)
    #   + RGB img
    train_x, train_y, test_x, test_y, classes = load_dataset()
    # check_img(25, train_x, train_y, classes)
    # Exercise_1(train_x, train_y, test_x, test_y)
    train_x, test_x = Exercise_2(train_x, train_y, test_x, test_y)
    
    # General Architecture of the learning algorithm
    # y_hat = sigmoid(W.T@x + B)
    # Loss_func = -y * log(y_hat) - (1 - y) * log(1 - y_hat)
    # Coss_func = 1/total_img * sum(Loss_func)

    # Main steps for neural network:
    #   + Define model architecture
    #   + Init parameter
    #   + Iteration process: Forward_prop -> Back_prop -> grad_des

    # Exercise_3()
    # Exercise_4()
    # Exercise_5()
    # Exercise_6()
    # Exercise_7()
    # logistic_regression_model = Exercise_8(train_x, train_y, test_x, test_y, classes, num_iterations=200000, print_cost=True)
    # learning_curve_plotting(logistic_regression_model)
    multiple_learning_testing(train_x, train_y, test_x, test_y, classes, num_iterations=5000, print_cost=True)
    return None


if __name__ == '__main__':
    main()
