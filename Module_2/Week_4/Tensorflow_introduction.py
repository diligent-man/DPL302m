import h5py
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable


###############################################################################
def read_dataset() -> tuple:
    train_set = h5py.File('train_signs.h5', 'r')
    test_set = h5py.File('test_signs.h5', 'r')
    
    x_train = tf.data.Dataset.from_tensor_slices(train_set['train_set_x'])
    y_train = tf.data.Dataset.from_tensor_slices(train_set['train_set_y'])
    
    x_test = tf.data.Dataset.from_tensor_slices(test_set['test_set_x'])
    y_test = tf.data.Dataset.from_tensor_slices(test_set['test_set_x'])
    return x_train, y_train, x_test, y_test


def dataset_inspecting(x_train, y_train) -> None:
    # Tensor dim check
    # Length
    print(f'''# of image in training set: {len(x_train)}
# of image in testing set: {len(y_train)}''')
    # Shape of 1 tensor
    print(f'Tensor dimension: {x_train.element_spec.shape}') 
    # Rank
    print(f'Tensor of rank: {tf.rank(tf.constant(next(iter(x_train)).numpy()))}')
    
    # Visualize
    # Since TensorFlow Datasets are generators, can't access directly unless iterate over it
    img = next(iter(x_train))
    plt.imshow(img.numpy().astype(np.uint8))
    plt.show()
    return None


def normalize(img):
    # Transform img into a tensor(64 * 64 * 3,) and normalize it
    img = img / 255
    img = tf.reshape(img, [-1]) # infered value must be tensor rank 1 not 0
    return img


################################################################################
# Linear func
def linear_function():
    W = tf.random.normal(shape=[4, 3], mean=0, stddev=1, name='weight')
    X = tf.random.normal(shape=[3, 1], mean=0, stddev=1, name='weight')
    b = tf.random.normal(shape=[4, 1], mean=0, stddev=1, name='weight')
    Y = tf.add(tf.matmul(W, X), b)
    return Y


def Exercise_1() -> None:
    # Task: Compute WX+b
    # where W,X,b are drawn from a rand normal dist.
    # Shape: W = (4, 3), X = (3,1), b = (4,1)
    Y = linear_function()
    return None


###############################################################################
def sigmoid(z):
    z = tf.cast(z, dtype=tf.float32)
    a = tf.sigmoid(z)
    return a


def Exercise_2() -> None:
    a = sigmoid([-1])
    b = sigmoid([0])
    c = sigmoid([1])
    d = sigmoid([10])
    print(a, b, c, d)
    return None


##############################################################################
def one_hot_matrix_test(target):
    label = tf.constant(1)
    C = 4
    result = target(label, C)
    print("Test 1:",result)
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 1. ,0., 0.] ), "Wrong output. Use tf.one_hot"
    label_2 = [2]
    result = target(label_2, C)
    print("Test 2:", result)
    print(result.shape[0])
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 0. ,1., 0.] ), "Wrong output. Use tf.reshape as instructed"
    
    print("\033[92mAll test passed")


def one_hot_matrix(label, C=6):
    one_hot = tf.one_hot(indices=label, depth=C)
    one_hot = tf.reshape(one_hot, shape=[-1,])
    return one_hot


def Exercise_3(labels):
    one_hot_matrix_test(one_hot_encoding)
    # labels = one_hot_matrix(labels)




def main() -> None:
    # check version
    # print(tf.__version__) # 2.13.0

    x_train, y_train, x_test, y_test = read_dataset()
    labels = np.unique([label.numpy() for label in y_train])
    # dataset_inspecting(x_train, y_train)

    # use map() in lieu of function call. Same as pd.apply() (64, 64, 3)
    x_train = x_train.map(normalize) # (1080, 64, 64, 3)
    x_test = x_test.map(normalize)   # (120, 12288)

    Exercise_1()
    # Exercise_2()
    # Exercise_3()

    y_test = y_test.map(one_hot_matrix)
    y_train = y_train.map(one_hot_matrix)

    print(next(iter(y_test)))

    return None


if __name__ == '__main__':
    main()