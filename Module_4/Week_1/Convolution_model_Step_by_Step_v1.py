from pprint import pprint as pp
import numpy as np
import h5py
import matplotlib.pyplot as plt


def zero_pad(X: np.ndarray, padding_size: int) -> np.ndarray:
    """
    Inp:
        X(m, n_C, n_H, n_W)
            m: # of images
            n_C: channels
            n_H: height
            n_W: width

    Out:
        X_pad: padded image
    """
    num_of_img = channel = (0, 0) 
    h = w = (padding_size, padding_size)
    pad_width = (num_of_img, h, w, channel)

    X_pad = np.pad(array=X, pad_width=pad_width,
                   mode='constant', constant_values=(0,0))
    return X_pad


def Exercise_1() -> None:
    np.random.seed(4)

    num_of_img = 3
    h = 2 ; w = 3; channel = 1
    x = np.random.randn(num_of_img, h, w, channel)
    x_pad = zero_pad(x, padding_size=1) 
    
    print ('First unpadded img\n{}\n'.format(x[0]))
    print ('First padded img\n{}\n'.format(x_pad[0]))

    
    # visualize
    fig, axarr = plt.subplots(3,2)
    for i in range(3):
        axarr[i][0].set_title(f'unpadded img {i}')
        axarr[i][0].imshow(x[i,], cmap='gray')

        axarr[i][1].set_title(f'padded img {i}')
        axarr[i][1].imshow(x_pad[i,], cmap='gray')
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    return None


#######################################################################
def conv_single_step(slider_tensor: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    """
    Inp
        prev_slice: result of previous convolution
        W: a filter
        b
        
    Out
        result of convolution operation
    """
    return np.sum(slider_tensor * W) + float(b)


def Exercise_2() -> None:
    np.random.seed(1)

    slider_tensor = np.random.randn(2, 2, 3) # (h, w, c)
    W = np.random.randn(2, 2, 3)             # (h, w, c)
    b = np.random.randn(1, 1, 1)             # (h, w, c)

    Z = conv_single_step(slider_tensor, W, b)
    return None


##########################################################################################
def conv_forward(prev_A: np.ndarray, W: np.ndarray, b: np.ndarray,
                 hyperparas: dict) -> np.ndarray:
    """
    Inp
        prev_A: output activations of the previous layer
            -> (m, n_H, n_W, n_C_prev)

        W: filter
            -> (f, f, prev_n_C, n_C)
            f: filter size
        b
            -> (1, 1, 1, n_C)
        hyperparas: dict for striding & padding_size
            
    Out:
    Z: stack of convolutions
        -> (m, n_H, n_W, n_C)
        n_C: accord with # of filters
        
        n_H = floor((prev_n_H - f + padding_size * 2) / stride) + 1
        
        n_W = floor((prev_n_W - f + padding_size * 2) / stride) + 1 

    cache -- cache of values needed for the conv_backward() function
    """
    # Retrieve dimensions from previous layer
    (m, prev_n_H, prev_n_W, prev_n_C) = prev_A.shape
    
    # Retrieve dimensions of filter
    (f, f, prev_n_C, n_C) = W.shape
    
    # Retrieve hyperparas
    stride = hyperparas["stride"]
    padding_size = hyperparas["padding_size"]
    
    # Out shape 
    n_H = ((prev_n_H - f + 2 * padding_size) // stride) + 1
    n_W = ((prev_n_W - f + 2 * padding_size) // stride) + 1                     

    # Initialize the out volume
    Z = np.zeros(shape=(m, n_H, n_W, n_C))
    
    # Padding prev_A by zero padding
    prev_padded_A = zero_pad(prev_A, padding_size=padding_size)

    # perform convolution operation
    for i in range(m):
        prev_padded_a = prev_padded_A[i]         # retrieve each image
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    upper_left = h * stride
                    lower_right = w * stride + f

                    # Calculate slider area
                    slider_tensor = prev_padded_a[upper_left: upper_left + f,
                                                  lower_right - f: lower_right, :]

                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    
                    Z[i, h, w, c] = conv_single_step(slider_tensor, weights ,biases)
    cache = (prev_A, W, b, hyperparas)
    return Z, cache


def Exercise_3() -> None:
    np.random.seed(1)
    
    # (m, n_H, n_W, n_C_prev)
    prev_A = np.random.randn(2, 5, 7, 4) # 2 img with size 5x7 & 4 channels
    
    # (f, f, prev_n_C, n_C)
    W = np.random.randn(3, 3, 4, 8) # 8 filters with size 3x3
    b = np.random.randn(1, 1, 1, 8) # 8 biases for i8 filters
    
    hyperparas = {"padding_size" : 1, "stride": 2}

    Z, cache_conv = conv_forward(prev_A, W, b, hyperparas)
    return None


#########################################################################################
def pool_forward(prev_A: np.ndarray, hyperparas: dict, mode = "max") -> tuple:
    """
    Inp
        prev_A: (m, n_H_prev, n_W_prev, n_C_prev)
        hyperparas: dict for filter_szie & stride
        mode: max | avg pooling
    Output
        A - (m, n_H, n_W, n_C)
        cache
    """
    (m, prev_n_H, prev_n_W, prev_n_C) = prev_A.shape
    f = hyperparas["f"]
    stride = hyperparas["stride"]
    
    # Define the dimensions of the output
    n_H = 1 + (prev_n_H - f) // stride
    n_W = 1 + (prev_n_W - f) // stride
    n_C = prev_n_C
    
    # Initialize output
    A = np.zeros((m, n_H, n_W, n_C))   

    # perform pooling
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    upper_left = h * stride
                    lower_right = w * stride + f

                    slider_tensor = prev_A[i, upper_left: upper_left + f,
                                              lower_right - f: lower_right, c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(slider_tensor)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(slider_tensor)
    
    cache = (prev_A, hyperparas)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache

def Exercise_4() -> None:
    print("CASE 1:")
    np.random.seed(1)
    hyperparas_case_1 = {"stride" : 1, "f": 3}
    prev_A_case_1 = np.random.randn(2, 5, 5, 3)
    print('Previous A case 1\n', prev_A_case_1)

    A, cache = pool_forward(prev_A_case_1, hyperparas_case_1, mode = "max")
    print("mode = max")
    print("A.shape = " + str(A.shape))
    print("A[0] =\n", A[0])
    
    A, cache = pool_forward(prev_A_case_1, hyperparas_case_1, mode = "average")
    print("mode = average")
    print("A.shape = " + str(A.shape))
    print("A[0] =\n", A[0])
    print()
    print()


    # Case 2: stride of 2
    print("\n\nCASE 2:")
    np.random.seed(1)
    hyperparas_case_2 = {"stride" : 2, "f": 3}
    prev_A_case_2 = np.random.randn(2, 5, 5, 3)
    print('Previous A case 2\n', prev_A_case_2)

    A, cache = pool_forward(prev_A_case_2, hyperparas_case_2, mode = "max")
    print("mode = max")
    print("A.shape = " + str(A.shape))
    print("A[0] =\n", A[0])
    print()

    A, cache = pool_forward(prev_A_case_2, hyperparas_case_2, mode = "average")
    print("mode = average")
    print("A.shape = " + str(A.shape))
    print("A[0] =\n", A[0])
    return None


######################################################################################
def conv_backward(dZ: np.ndarray, cache: list) -> tuple:
    """
    Inp
        dZ: grad of cost w.r.t output of conv layer Z,
            -> (m, n_H, n_W, n_C)
        cache: output from conv_forward of Z
        
    Out
        prev_dA: grad of cost w.r.t prev_A
            -> (m, prev_n_H, prev_n_W, prev_n_C)

        dW: (f, f, prev_n_C, n_C)
        db: (1, 1, 1, n_C)
    """
    prev_A, W, b, hyperparas = cache
    stride = hyperparas["stride"]
    padding_size = hyperparas["padding_size"]

    m, prev_n_H, prev_n_W, prev_n_C = prev_A.shape
    f, f, prev_n_C, n_C = W.shape
    m, n_H, n_W, n_C = dZ.shape
    
    # Initialize prev_dA, dW, db with the correct shapes
    prev_dA = np.zeros((m, prev_n_H, prev_n_W, prev_n_C))                          
    dW = np.zeros((f, f, prev_n_C, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    # Padding
    prev_padded_A = zero_pad(prev_A, padding_size)
    prev_padded_dA = zero_pad(prev_dA, padding_size)

    # perform backprop
    for i in range(m):
        prev_padded_a = prev_padded_A[i]
        prev_padded_da = prev_padded_dA[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    upper_left = h * stride
                    lower_right = w * stride + f

                    slider_tensor = prev_padded_a[upper_left : upper_left + f,
                                                  lower_right - f : lower_right,
                                                  :]

                    # Update grads for window and the filter's paras
                    prev_padded_da[upper_left: upper_left + f,
                                   lower_right - f: lower_right, :] +=\
                    W[:, :, :, c] * dZ[i, h, w, c]
                    
                    dW[:, :, :, c] += slider_tensor * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Set i_th example's prev_dA to previous tensor
        prev_dA[i] = prev_padded_da[padding_size: -padding_size,
                                    padding_size: -padding_size, :]
    return prev_dA, dW, db


def Exercise_5() -> None:
    np.random.seed(1)
    # contrived init
    prev_A = np.random.randn(1, 2, 2, 1)
    W = np.random.randn(2, 2, 1, 2)
    b = np.random.randn(1, 1, 1, 2)
    hyperparas = {"padding_size" : 1, "stride": 1}
    
    # feed forward
    Z, cache = conv_forward(prev_A, W, b, hyperparas)

    # backprop
    dA, dW, db = conv_backward(Z, cache)
    return None


################################################################################
def create_mask_from_window(x):
    """
    Inp: x - (f, f)

    Out: mask - arr contains pos of max val
    """
    return x == np.max(x)


def Exercise_6() -> None:
    np.random.seed(1)
    x = np.random.randn(2, 3)
    mask = create_mask_from_window(x)
    
    print('x=')
    pp(x)

    print("mask=")
    pp(mask)
    return None


################################################################################
def distribute_value(dz, filter_shape):
    """
    This func is used for avg pooling that equally spreads influence of surrounding values

    Inp
        dz: inp scalar
        shape: which we want to distribute the value of dz
            -> (n_H, n_W)
    
    Out
        a: which we distributed the value of dz
            -> (n_H, n_W)
    """
    n_H, n_W = filter_shape
    average = np.ones(filter_shape) / np.multiply(n_H, n_W)
    return dz * average


def Exercise_7() -> None:
    a = distribute_value(2, (2, 2))
    print('distributed value =')
    pp(a)
    return None


################################################################################
def pool_backward(dA, cache, mode = "max"):
    """
    Inp
        dA: grad of w.r.t output of the pooling layer
        cache
        mode: max | average
    
    Out
        prev_dA: grad of cost w.r.t input pooling layer, same shape as prev_A
    """
    prev_A, hyperparas = cache
    stride = hyperparas["stride"]
    f = hyperparas["f"]
    
    m, prev_n_H, prev_n_W, prev_n_C = prev_A.shape
    m, n_H, n_W, n_C = dA.shape
    
    prev_dA = np.zeros(prev_A.shape)

    for i in range(m):
        prev_a = prev_A[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    upper_left = h * stride
                    lower_right = w * stride + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        slider_tensor = prev_a[upper_left: upper_left + f,
                                               lower_right - f: lower_right, c]

                        mask = create_mask_from_window(slider_tensor)

                        prev_dA[i, upper_left: upper_left + f
                                 , lower_right - f: lower_right, c] += dA[i, h, w, c] * mask
                        
                    elif mode == "average":
                        da = dA[i, h, w, c]

                        prev_dA[i, upper_left: upper_left + f,
                                   lower_right - f: lower_right, c] += distribute_value(da, (f, f))
    return prev_dA


def Exercise_8() -> None:
    np.random.seed(1)

    # contrived init
    prev_A = np.random.randn(5, 5, 3, 2)
    dA = np.random.randn(5, 4, 2, 2)
    
    hyperparas = {"stride" : 1, "f": 2}
    A, cache = pool_forward(prev_A, hyperparas)
    

    # Test backprop max pooling    
    dA_prev1 = pool_backward(dA, cache, mode = "max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev1[1,1] = ', dA_prev1[1, 1])  
    print()


    # Test backprop avg pooling
    dA_prev2 = pool_backward(dA, cache, mode = "average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev2[1,1] = ', dA_prev2[1, 1])
    return None


################################################################################
def main() -> None:
    """
    Convolution funcs:
        + Zero padding
        + Convolve window/ slider
        + Feed forward
        + Backprop

    Pooling funcs:
        + Forward pooling
        + Create mask
        + Distribute value
        + Backward pooling
    This assignment will be implemented by numpy
    """

    # Exercise_1()
    # Exercise_2()
    # Exercise_3()
    # Exercise_4()
    # Exercise_5()
    # Exercise_6()
    # Exercise_7()
    # Exercise_8()
    return None


if __name__ == '__main__':
    main()