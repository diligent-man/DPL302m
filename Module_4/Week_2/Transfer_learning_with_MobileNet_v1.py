import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras as kr
from pprint import pprint as pp


def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    data_augmenter = kr.Sequential()
    data_augmenter.add(kr.layers.RandomFlip(mode='horizontal'))
    data_augmenter.add(kr.layers.RandomRotation(factor=1, interpolation='bilinear'))
    return data_augmenter


def visualize_data_set(class_names: str, train_set) -> None:
    plt.figure(figsize=(10, 10))
    for images, labels in train_set.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()
    return None


def visualize_augmented_data(train_set, data_augmentation) -> None:
    for image, labels in train_set.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            # tf.expand_dims: Add an outer "batch" dimension to a single element.
            # e.g: inp=(160, 160, 3) -> inp=(1, 160, 160, 3)
            augmented_image = data_augmentation(tf.expand_dims(first_image, axis=0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')
    plt.show()
    return None

def MobileNetV2(input_shape):
    return kr.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=True,            # whether to include the fully-connected layer at the top of the network
                                                   weights='imagenet',          # pretrained weights
                                                   alpha=1.0,                   # # of filters from the paper are used at each layer.
                                                   pooling='max',
                                                   classes=1000,
                                                   classifier_activation='softmax')

def MobileNetV3Large(input_shape):
    return kr.applications.MobileNetV3Large(input_shape=input_shape,
                                                   include_top=True,            # whether to include the fully-connected layer at the top of the network
                                                   weights='imagenet',          # pretrained weights
                                                   alpha=1.0,                   # # of filters from the paper are used at each layer.
                                                   pooling='max',
                                                   classes=1000,
                                                   classifier_activation='softmax')


def test_pretrained(pretrained_model, train_set) -> None:
    # Num of layers in model:
    num_of_layers = len(pretrained_model.layers)
    print('Num of layers:', num_of_layers)

    # Retrieve first batch
    mini_batch, label_batch = next(iter(train_set))


    # preprocessing data (scale input pixels between -1 and 1) before passing through model
    inputs = tf.Variable(kr.applications.mobilenet_v2.preprocess_input(mini_batch))
    predictions = pretrained_model(inputs)

    # , then a human-readable label, and last the probability of the image belonging to that class. You'll notice that there are two of these returned for each image in the batch - these the top two probabilities returned for that image.
    for result in kr.applications.mobilenet_v2.decode_predictions(preds=predictions.numpy(), top=5):
        for class_number, class_name, probability in result:
            print(f"""Class number: {class_number},
Class name: {class_name},
Probability: {probability}\n
""")

    return None


####################################################################################################################
def alpaca_model(img_size, data_augmenter=data_augmenter()):
    '''
    Define a model for binary classification based on the MobileNetV2 model

    Inp
        image_shape
        data_augmentation

    Out
        tf.keras.model
    '''

    # Tweaking pretrained model steps:
    #     S1: Delete the top layer (the classification layer)
    #         -> Set include_top in base_model as False
    #
    #     S2: Add a new classifier layer
    #     -> Train only one layer by freezing the rest of the network (Single neuron is enough to solve a binary classification problem)
    #
    #     S3: Freeze the base model and train the newly-created classifier layer
    #         -> Set base model.trainable=False to avoid changing the weights and train only the new layer
    #         -> Set training in base_model to False to avoid keeping track of statistics in the batch norm layer
    input_shape = img_size + (3,)

    # S1
    model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                              include_top=False,  # <== Important!!!!
                                              weights='imagenet') # From imageNet
    model.trainable = False # freeze the base model by making it non-trainable

    # S2
    inputs = tf.keras.Input(shape=input_shape)
    
    # data augmentation
    x = data_augmenter(inputs)
    x = kr.applications.mobilenet_v2.preprocess_input(x)


    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = model(x, training=False)

    # S3: add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = kr.layers.GlobalAveragePooling2D(keepdims=False)(x)
    x = kr.layers.Dropout(0.2)(x)
    outputs = kr.layers.Dense(1)(x) # use a prediction layer with one neuron (as a binary classifier only needs one)
    return kr.Model(inputs=inputs, outputs=outputs)


###########################################################################################################################
def visualize_result(history) -> None:
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    return None


#####################################################################################################################
def main() -> None:
    batch_size = 1; img_size = (160, 160); dir = "dataset"
    train_set = kr.preprocessing.image_dataset_from_directory(directory=dir, shuffle=True,
                                                              batch_size=batch_size, image_size=img_size,
                                                              validation_split=0.2, subset='training',  seed=1)
    class_names = train_set.class_names
    # prevent memory bottlenecks when reading from disk
    train_dataset = train_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    dev_set = kr.preprocessing.image_dataset_from_directory(directory=dir, shuffle=True,
                                                            batch_size=batch_size, image_size=img_size,
                                                            validation_split=0.2, subset='validation', seed=1)
    
    
    
    # visualize_train_set()
    data_augmentation = data_augmenter()
    # visualize_augmented_data(train_set, data_augmentation)


    # tf.keras.applications: pretrained models and relevant stuff
    # Note: Note the last 2 layers here.
    # They are the so called top layers, and they are responsible for the classification in the model
    # Instantiate MobileNetV2 model
    input_shape = img_size + (3,) # (160, 160, 3)
    pretrained_MobileNetV2 = MobileNetV2(input_shape)
    # pretrained_MobileNetV2.summary()
    # test_pretrained(pretrained_MobileNetV2, train_dataset)


    # Instantiate MobileNetV3 model
    pretrained_MobileNetV3Large = MobileNetV3Large(input_shape)
    # pretrained_MobileNetV3Large.summary()
    # test_pretrained(pretrained_MobileNetV3Large, train_dataset)


    ######################################################################################################
    tweaked_pretrained_MobileNetV2 = alpaca_model(img_size, data_augmentation)
    tweaked_pretrained_MobileNetV2.summary()


    tweaked_pretrained_MobileNetV2.compile(optimizer=kr.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, jit_compile=True),
                                              loss=kr.losses.BinaryCrossentropy(from_logits=True),
                                              metrics=['accuracy', 'mse'])
    # history = tweaked_pretrained_MobileNetV2.fit(train_dataset, validation_data=dev_set, epochs=40)
    # visualize_result(history)
    return None

if __name__ == '__main__':
    main()