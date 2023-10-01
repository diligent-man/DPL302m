import numpy as np
import math

##################################################################################
def Exercise_1() -> None:
    test = "Hello World"
    print(test)
    return None


##################################################################################
def basic_sigmoid(x: float) -> float:
    # Basic sigmoid func
    # Formula: 1 / (1 + e^(-x))   
    s = 1 / (1 + math.exp(-x))
    return s


def Excersise_2() -> None:
    test_1 = basic_sigmoid(0)
    test_2 = basic_sigmoid(1)
    print(test_1, test_1)

    # Raise error cuz math does not work with list of nums
    # x = np.array([1,2,3])
    # y = basic_sigmoid(x)
    return None


########################################################################################
def sigmoid(x: list) -> list:
    s = 1 / (1 + np.exp(-x))
    return s


def Exercise_3() -> None:
    t_x = np.array([1, 2, 3])
    print("sigmoid(t_x) = " + str(sigmoid(t_x)))
    return None


#####################################################################################
def sigmoid_derivative(x: list) -> list:
    # Sigmoid mathematical denotation: sigma
    # Sigmoid derivative: sigma_prime = sigma * (1 - sigma)
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


def Exercise_4() -> None:
    t_x = np.array([1, 2, 3])
    print ("sigmoid_derivative(t_x) = " + str(sigmoid_derivative(t_x)))
    return None


#####################################################################################
def img2vec(img: list) -> list:
    # This is also called flatten operation

    # Vector with size n * 0 <==> lst
    # v = img.reshape(img.shape[0] * img.shape[1] * img.shape[2])
    # equivalent to img.reshape(-1)

    v = img.reshape(-1,1) # vector with size n * 1
    return v


def Exercise_5() -> None:
    t_image = np.array([[[0.67826139, 0.29380381],
                         [0.90714982, 0.52835647],
                         [0.4215251 , 0.45017551]],

                        [[ 0.92814219, 0.96677647],
                         [ 0.85304703, 0.52351845],
                         [ 0.19981397, 0.27417313]],

                        [[0.60659855, 0.00533165],
                         [0.10820313, 0.49978937],
                         [0.34144279, 0.94630077]]]
                         )
    print("image2vector(image) = " + str(img2vec(t_image)))
    return None


############################################################################
def normalize_rows(x: list) -> list:
    # Compute L2-norm

    # Way 1
    x_norm = np.linalg.norm(x=x, axis=1, keepdims=True)
    x = np.divide(x, x_norm)
    
    # Way 2
    # x = np.absolute(x)
    return x


def Exercise_6() -> None:
    # x is currently deemed as a row vec
    x = np.array([[0, 3, 4],
                  [1, 6, 4]])
    print("normalizeRows(x) = " + str(normalize_rows(x)))
    return None


##########################################################################
def softmax(x: list) -> list:
    # row vec
    # exp of each entries
    x_exp = np.exp(x)
    # sum of each rows
    x_sum = np.sum(x_exp, axis=1 , keepdims=True)
    # softmax
    s = np.divide(x_exp, x_sum)
    return s


def Exercise_7() -> None:
    t_x = np.array([[9, 2, 5, 0, 0],
                    [7, 5, 0, 0 ,0]])
    print("softmax(x) = " + str(softmax(t_x)))
    return None


################################################################################
def L1(yhat: list, y:list) -> list:
    # Formula: L1 = sigma(y_hat - y)
    loss = np.sum(np.absolute(np.subtract(y, yhat)))
    return loss

def Exercise_8() -> None:
    # Implement L1-loss
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(L1(yhat, y)))
    return None


################################################################################
def L2(yhat: list, y:list) -> list:
    # Formula: L2 = sigma(np.power(y_hat - y), 2)
    loss = np.sum(np.power(np.subtract(y, yhat), 2))
    return loss

def Exercise_9() -> None:
    # Implement L2-loss
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(L1(yhat, y)))
    return None



def main() -> None:
    # Exercise_1()
    # Exersise_2()
    # Exercise_3()
    # Exercise_4()
    # Exercise_5()
    # Exercise_6()
    # Exercise_7()
    # Exercise_8()
    # Exercise_9()


if __name__ == '__main__':
    main()