import h5py
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
    y_test = tf.data.Dataset.from_tensor_slices(test_set['test_set_y'])
    return x_train, y_train, x_test, y_test


def dataset_inspecting(x_train, y_train) -> None:
    # Tensor dim check
    # Length
    print(f'''# of image in training set: {len(x_train)}
# of image in testing set: {len(y_train)}''')
    # Shape of 1 tensor
    print(f'Image dimension: {x_train.element_spec.shape}')

    # Visualize
    # Since TensorFlow Datasets are generators, can't access directly unless iterate over it
    img = next(iter(x_train))
    plt.imshow(img.numpy().astype(np.uint8))
    plt.show()
    return None


def normalize(img):
    # Transform img into a tensor(64 * 64 * 3,) and normalize it
    img = img / 255
    img = tf.reshape(img, [-1])  # infered value must be tensor rank 1 not 0
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
    print("Test 1:", result)
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 1., 0., 0.]), "Wrong output. Use tf.one_hot"
    label_2 = [2]
    result = target(label_2, C)
    print("Test 2:", result)
    print(result.shape[0])
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 0., 1., 0.]), "Wrong output. Use tf.reshape as instructed"
    
    print("\033[92mAll test passed")


def one_hot_matrix(labels, C=6):
    one_hot = tf.one_hot(indices=labels, depth=C)
    one_hot = tf.reshape(one_hot, shape=[-1, ])
    return one_hot


def Exercise_3(labels):
    one_hot_matrix_test(one_hot_matrix)


############################################################################################
def initialize_parameters_test(target):
    parameters = target()

    values = {"W1": (25, 12288),
              "b1": (25, 1),
              "W2": (12, 25),
              "b2": (12, 1),
              "W3": (6, 12),
              "b3": (6, 1)}

    for key in parameters:
        print(f"{key} shape: {tuple(parameters[key].shape)}")
        assert type(parameters[key]) == ResourceVariable, "All parameter must be created using tf.Variable"
        assert tuple(parameters[key].shape) == values[key], f"{key}: wrong shape"
        assert np.abs(np.mean(parameters[key].numpy())) < 0.5, f"{key}: Use the GlorotNormal initializer"
        assert np.std(parameters[key].numpy()) > 0 and np.std(
            parameters[key].numpy()) < 1, f"{key}: Use the GlorotNormal initializer"
    print("\033[92mAll test passed")

def initialize_parameters():
    """
    Shape:
    W1 : [25, 12288]
    b1 : [25, 1]
    W2 : [12, 25]
    b2 : [12, 1]
    W3 : [6, 12]
    b3 : [6, 1]
    """

    initializer = tf.keras.initializers.GlorotNormal(seed=1)

    W1 = tf.Variable(initial_value=initializer(shape=(25, 12288)))
    b1 = tf.Variable(initial_value=initializer(shape=(25, 1)))
    W2 = tf.Variable(initial_value=initializer(shape=(12, 25)))
    b2 = tf.Variable(initial_value=initializer(shape=(12, 1)))
    W3 = tf.Variable(initial_value=initializer(shape=(6, 12)))
    b3 = tf.Variable(initial_value=initializer(shape=(6, 1)))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters


def Exercise_4():
    initialize_parameters_test(initialize_parameters)


#############################################################################################
def forward_propagation_test(target, examples):
    minibatches = examples.batch(2)
    parametersk = initialize_parameters()
    W1 = parametersk['W1']
    b1 = parametersk['b1']
    W2 = parametersk['W2']
    b2 = parametersk['b2']
    W3 = parametersk['W3']
    b3 = parametersk['b3']
    index = 0
    minibatch = list(minibatches)[0]
    with tf.GradientTape() as tape:
        forward_pass = target(tf.transpose(minibatch), parametersk)
        print(forward_pass)
        fake_cost = tf.reduce_mean(forward_pass - np.ones((6, 2)))

        assert type(forward_pass) == EagerTensor, "Your output is not a tensor"
        assert forward_pass.shape == (6, 2), "Last layer must use W3 and b3"
        assert np.allclose(forward_pass,
                           [[-0.13430887,  0.14086473],
                              [0.21588647, -0.02582335],
                              [0.7059658,   0.6484556],
                              [-1.1260961,  -0.9329492],
                              [-0.20181894, -0.3382722],
                              [0.9558965,   0.94167566]
                            ]), "Output does not match"
    index = index + 1
    trainable_variables = [W1, b1, W2, b2, W3, b3]
    grads = tape.gradient(fake_cost, trainable_variables)
    assert not(None in grads), "Wrong gradients. It could be due to the use of tf.Variable whithin forward_propagation"
    print("\033[92mAll test passed")


def forward_propagation(X, parameters):
    # Arch: ReLU(LINEAR) -> ReLU(LINEAR) -> LINEAR
    # X: (input size, number of examples)

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']; b1 = parameters['b1']
    W2 = parameters['W2']; b2 = parameters['b2']
    W3 = parameters['W3']; b3 = parameters['b3']

    # Feed forward
    Z1 = tf.add(W1 @ X, b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.add(W2 @ A1, b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.add(W3 @ A2, b3)
    return Z3

def Exercise_5(x_train):
    forward_propagation_test(forward_propagation, x_train)


##########################################################################################
def compute_total_loss_test(target, Y):
    pred = tf.constant([[2.4048107, 5.0334096],
                        [-0.7921977, -4.1523376],
                        [0.9447198, -0.46802214],
                        [1.158121, 3.9810789],
                        [4.768706, 2.3220146],
                        [6.1481323, 3.909829]])
    minibatches = Y.batch(2)
    for minibatch in minibatches:
        result = target(pred, tf.transpose(minibatch))
        break

    print(result)
    assert (type(result) == EagerTensor), "Use the TensorFlow API"
    assert (np.abs(result - (0.50722074 + 1.1133534) / 2.0) < 1e-7), "Test does not match. Did you get the reduce sum of your loss functions?"

    print("\033[92mAll test passed")


def compute_total_loss(logits, labels):
    # logits: (-1, # of examples)
    # labels: (-1, # of examples)

    y_true = tf.transpose(labels)
    y_pred = tf.transpose(logits)
    total_loss = tf.keras.metrics.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)
    total_loss = tf.reduce_sum(total_loss)
    return total_loss


def Exercise_6(y_train):
    compute_total_loss_test(compute_total_loss, y_train)


##########################################################################################
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []                                        # To keep track of the cost
    train_acc = []
    test_acc = []

    # Initialize your parameters
    #(1 line)
    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # The CategoricalAccuracy will track the accuracy for this multiclass problem
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()

    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))

    # We can get the number of elements of a dataset using the cardinality method
    m = dataset.cardinality().numpy()

    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)
    # X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)# <<< extra step
    # Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8) # loads memory faster

    # Do the training loop
    for epoch in range(num_epochs):

        epoch_total_loss = 0.

        # We need to reset object to start measuring from 0 the accuracy each epoch
        train_accuracy.reset_states()

        for (minibatch_X, minibatch_Y) in minibatches:

            with tf.GradientTape() as tape:
                # 1. predict
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)

                # 2. loss
                minibatch_total_loss = compute_total_loss(Z3, tf.transpose(minibatch_Y))

            # We accumulate the accuracy of all the batches
            train_accuracy.update_state(minibatch_Y, tf.transpose(Z3))

            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_total_loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_total_loss += minibatch_total_loss

        # We divide the epoch total loss over the number of samples
        epoch_total_loss /= m

        # Print the cost every 10 epochs
        if print_cost is True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
            print("Train accuracy:", train_accuracy.result().numpy())

            # We evaluate the test set every 10 epochs to avoid computational overhead
            for (minibatch_X, minibatch_Y) in test_minibatches:
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                test_accuracy.update_state(minibatch_Y, tf.transpose(Z3))
            print("Test_accuracy:", test_accuracy.result().numpy())

            costs.append(epoch_total_loss)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()
            print()

    return parameters, costs, train_acc, test_acc

def Exercise_7(x_train, y_train, x_test, y_test, num_epochs=100):
    parameters, costs, train_acc, test_acc = model(x_train, y_train,
                                                   x_test, y_test,
                                                   num_epochs=num_epochs)
    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))
    plt.show()

    # Plot the train accuracy
    plt.plot(np.squeeze(train_acc))
    plt.ylabel('Train Accuracy')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))
    # Plot the test accuracy
    plt.plot(np.squeeze(test_acc))
    plt.ylabel('Test Accuracy')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))
    plt.show()


##########################################################################################
def main() -> None:
    # check version
    # print(tf.__version__) # 2.13.0

    x_train, y_train, x_test, y_test = read_dataset()
    labels = np.unique([label.numpy() for label in y_train])
    # dataset_inspecting(x_train, y_train)

    # use map() in lieu of function call. Same as pd.apply() (64, 64, 3)
    x_train = x_train.map(normalize) # (1080, 64, 64, 3)
    x_test = x_test.map(normalize)   # (120, 12288)

    # Exercise_1()
    # Exercise_2()
    # Exercise_3(labels)

    y_train = y_train.map(one_hot_matrix)
    y_test = y_test.map(one_hot_matrix)
    parameters = initialize_parameters()

    # Exercise_4()
    # Exercise_5(x_train)
    # Exercise_6(y_train)
    Exercise_7(x_train, y_train, x_test, y_test)
    return None


if __name__ == '__main__':
    main()