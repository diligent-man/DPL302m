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


#######################################################################
def conv_forward(prev_A: np.ndarray, W: np.ndarray, b: np.ndarray,
                 hyperparas:dict) -> np.ndarray:
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
    print(n_H, n_W)                                                                                                                                                                                                                                             
    # Initialize the out volume
    Z = np.zeros(shape=(m, n_H, n_W, n_C))
    
    # Padding prev_A by zero padding
    prev_padded_A = zero_pad(prev_A, padding_size=padding_size)

    # perform convolution operation
    for i in range(m):
        # retrieve each image
        prev_padded_a = prev_padded_A[i] 

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Calculate slider area
                    slider_height = h * stride: h * stride + f
                    slider_width = w * stride: w * stride + f
                    slider_tensor = prev_padded_a[slider_height, slider_width, :]

                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    
                    Z[i, h, w, c] = conv_single_step(slider_tensor, weights ,biases)
    return Z


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


#######################################################################
def pool_forward(prev_A: np.ndarray, hyperparas: dict, mode = "max") -> tuple:

    return A, cache


def Exercise_4() -> None:

    return None


#######################################################################
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
    Exercise_4()
    return None


if __name__ == '__main__':
    main()