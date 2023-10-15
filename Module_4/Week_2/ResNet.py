import numpy as np
from tensorflow import keras as kr
from resnets_utils import load_dataset, OHE



def identity_block(X: np.ndarray, filter_size: int, filters: list, initializer=kr.initializers.random_uniform):
    """
    Inp
        X: minibatch from prev layer
            (m, n_H_prev, n_W_prev, n_C_prev)
        f: kernel size
        filters: # filters in main path
        initializer: random uniform initializer
    
    Out
    X: identity block
        (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    Filter1, Filter2, Filter3 = filters
    
    # Cache input for later addition
    X_shortcut = X
    
    # First component of main path
    # padding = 'valid' <==> no padding
    # filters: dimensionality of the output space
    X = kr.layers.Conv2D(filters=Filter1, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = kr.layers.BatchNormalization()(X) # along channels
    X = kr.layers.Activation('relu')(X)

    # Second component of main path
    # padding = 'same' + stride = 1 => retain resolution
    X = kr.layers.Conv2D(filters=Filter2, kernel_size=filter_size, strides=1, padding='same', kernel_initializer=initializer(seed=0))(X)
    X = kr.layers.BatchNormalization()(X) # Default axis
    X = kr.layers.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    # padding = 'valid' for downsampling
    X = kr.layers.Conv2D(filters=Filter3, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = kr.layers.BatchNormalization()(X)  # along channels
    
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = kr.layers.Add()([X, X_shortcut])
    X = kr.layers.Activation('relu')(X)
    return X


#########################################################################################################
def convolutional_block(X: np.ndarray, filter_size: int, filters: list, stride: int=2,
                        initializer=kr.initializers.glorot_uniform):
    """
    Inp
    X - (m, n_H_prev, n_W_prev, n_C_prev)
    filter_size
    filters: # the number of filters in main path
    stride
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer,
                   also called Xavier uniform initializer.

    Out
    X - (m, n_H, n_W, n_C)
    """
    # Retrieve Filters
    Filter1, Filter2, Filter3 = filters

    # Cache input for later addition
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    # padding = 'valid' <==> no padding
    # filters: dimensionality of the output space
    X = kr.layers.Conv2D(filters=Filter1, kernel_size=1, strides=stride, padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = kr.layers.BatchNormalization()(X) # along channels
    X = kr.layers.Activation('relu')(X)

    # Second component of main path
    # padding = 'same' + stride = 1 => retain resolution
    X = kr.layers.Conv2D(filters=Filter2, kernel_size=filter_size, strides=1, padding='same', kernel_initializer=initializer(seed=0))(X)
    X = kr.layers.BatchNormalization()(X) # along channels
    X = kr.layers.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    # padding = 'valid' for downsampling
    X = kr.layers.Conv2D(filters=Filter3, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = kr.layers.BatchNormalization()(X)  # along channels


    ##### SHORTCUT PATH #####
    X_shortcut = kr.layers.Conv2D(filters=Filter3, kernel_size=1, strides=stride, padding='valid', kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = kr.layers.BatchNormalization()(X_shortcut)

    # Final step: Add shortcut value to main path
    X = kr.layers.Add()([X, X_shortcut])
    X = kr.layers.Activation('relu')(X)
    return X


########################################################################################################
def ResNet50(input_shape: tuple, classes: int):
    """
                                                        ResNet50 arch
    Zero-padding pads the input with a pad of (3,3)

    Stage 1:
        2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2).
        BatchNorm
        MaxPooling uses a (3,3) window and a (2,2) stride.

    Stage 2:
        The convolutional block uses 3 sets of filters of size [64,64,256], "filter_size" is 3, and "stride" is 1.
        The 2 identity blocks use 3 sets of filters of size [64,64,256], and "filter_size" is 3.

    Stage 3:
        The convolutional block uses three sets of filters of size [128,128,512], "filter_size" is 3 and "stride" is 2.
        The 3 identity blocks use three sets of filters of size [128,128,512] and "filter_size" is 3.

    Stage 4:
        The convolutional block uses three sets of filters of size [256, 256, 1024], "filter_size" is 3 and "stride" is 2.
        The 5 identity blocks use three sets of filters of size [256, 256, 1024] and "filter_size" is 3.

    Stage 5:
        The convolutional block uses three sets of filters of size [512, 512, 2048], "filter_size" is 3 and "stride" is 2.
        The 2 identity blocks use three sets of filters of size [512, 512, 2048] and "filter_size" is 3.
        AvgPooling uses a window of shape (2,2).

        The 'flatten' layer doesn't have any hyperparameters.
        The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation.


    Stage-wise implementation:
        ReLU(BatchNorm(CONV2D)) -> MaxPool ->
        ConvBlock -> IDBlock * 2 ->
        ConvBlock -> IDBlock * 3 ->
        ConvBlock -> IDBlock * 5 ->
        ConvBlock -> IDBlock * 2 ->
        AvgPooL -> Flatten -> Densse -> Softmax
    * IDBlock == identity_block


    Inp:
    input_shape
    classes

    Returns:
    model
    """

    # Define the input as a tensor with shape input_shape
    inputs = kr.layers.Input(shape=input_shape)

    # Zero-Padding
    X = kr.layers.ZeroPadding2D(padding=(3, 3))(inputs)

    # Stage 1
    X = kr.layers.Conv2D(filters=64, kernel_size=7, strides=2, kernel_initializer=kr.initializers.glorot_uniform(seed=0))(X)
    X = kr.layers.BatchNormalization()(X)
    X = kr.layers.Activation('relu')(X)
    X = kr.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, filter_size=3, filters=[64, 64, 256], stride=1)
    X = identity_block(X, filter_size=3, filters=[64, 64, 256])
    X = identity_block(X, filter_size=3, filters=[64, 64, 256])

    # Stage 3
    X = convolutional_block(X, filter_size=3, filters=[128, 128, 512], stride=2)
    X = identity_block(X, filter_size=3, filters=[128, 128, 512])
    X = identity_block(X, filter_size=3, filters=[128, 128, 512])
    X = identity_block(X, filter_size=3, filters=[128, 128, 512])

    # Stage 4
    X = convolutional_block(X, filter_size=3, filters=[256, 256, 1024], stride=2)
    X = identity_block(X, filter_size=3, filters=[256, 256, 1024])
    X = identity_block(X, filter_size=3, filters=[256, 256, 1024])
    X = identity_block(X, filter_size=3, filters=[256, 256, 1024])
    X = identity_block(X, filter_size=3, filters=[256, 256, 1024])
    X = identity_block(X, filter_size=3, filters=[256, 256, 1024])

    ## Stage 5 (≈3 lines)
    X = convolutional_block(X, filter_size=3, filters=[512, 512, 2048], stride=2)
    X = identity_block(X, filter_size=3, filters=[512, 512, 2048])
    X = identity_block(X, filter_size=3, filters=[512, 512, 2048])

    ## AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X =kr.layers.AveragePooling2D()(X)

    # output layer
    X = kr.layers.Flatten()(X)
    outputs = kr.layers.Dense(classes, activation='softmax', kernel_initializer=kr.initializers.glorot_uniform(seed=0))(X)

    # Create model
    model = kr.Model(inputs=inputs, outputs=outputs)
    return model


#########################################################################################################
def preprocessing(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    x_train = x_train / 255.
    x_test = x_test / 255.

    # One hot encoding
    y_train = OHE(y_train, 6).T
    y_test = OHE(y_test, 6).T
    return x_train, y_train, x_test, y_test

def dataset_inspection(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    print("number of training examples = " + str(x_train.shape[0]))
    print("number of test examples = " + str(x_test.shape[0]))
    print("X_train shape: " + str(x_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(x_test.shape))
    print("Y_test shape: " + str(y_test.shape))
    return None

def main() -> None:
    """"
    1/ Identity_block arch
    X ---> ReLU(BatchNorm(Conv2D)) ---> ReLU(BatchNorm(Conv2D)) ---> BatchNorm(Conv2D)--->  +  ReLU --->
       |                                                                                    |
       |                                                                                    |
       |-------------------------------------------------------------------------------------
    -> identity_block func

    Remark: Initial X may be incompatible with X after passed through main path,
            we therefore apply conv for initial  X in shortcut path to make its volume compatible

    2/ Identity block with convolution in shortcut path
    X ---> ReLU(BatchNorm(Conv2D)) ---> ReLU(BatchNorm(Conv2D)) ---> BatchNorm(Conv2D)--->  +  ReLU --->
       |                                                                                    |
       |                                                                                    |
       |-------------------------------> BatchNorm(Conv2D) ----------------------------------
    -> convolutional_block func
    """
    x_train, y_train, x_test, y_test, classes = load_dataset(train_filename='train_signs.h5', test_filename='test_signs.h5')
    x_train, y_train, x_test, y_test = preprocessing(x_train, y_train, x_test, y_test)
    dataset_inspection(x_train, y_train, x_test, y_test)

    model = ResNet50(input_shape=x_train[0].shape, classes=y_train.shape[1])
    model.summary()

    optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'], jit_compile=True)

    model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, verbose=1,
               validation_data=(x_train, y_train), validation_split=0.2, shuffle=True,
               workers=4, use_multiprocessing=True)

    # save paras
    model.save_weights(save_format='h5', filepath="Pretrained_ResNet50.h5", overwrite=True)
    return None

if __name__ == '__main__':
    main()

