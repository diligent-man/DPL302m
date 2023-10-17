#!/usr/bin/env python
# coding: utf-8

#  #using transfer learning on a pre-trained CNN to build an Alpaca/Not Alpaca classifier!
# 

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation


# In[ ]:


BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "C:/Users/Thien An/Desktop/Module 4 - Week 2 + 3/archive/dataset"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)


# In[ ]:


class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# In[ ]:


def data_augmenter():# tạo ra nhiều phiên bản biến đổi của dữ liệu huấn luyện 
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip("horizontal"))
    data_augmentation.add(RandomRotation(0.2))
    return data_augmentation


# In[ ]:


data_augmentation = data_augmenter()

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')


# In[ ]:


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


# In[ ]:


IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=True,
                                               weights='imagenet')#trọng số đã được huấn luyện trước từ tập dữ liệu ImageNet


# In[ ]:


base_model.summary()


# In[ ]:


image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)


# In[ ]:


base_model.trainable = False #tránh thay đổi trọng số và chỉ cập nhật lớp mới
image_var = tf.Variable(image_batch) #tạo một biến TensorFlow để chứa batch hình ảnh 
pred = base_model(image_var)# thực hiện dự đoán

tf.keras.applications.mobilenet_v2.decode_predictions(pred.numpy(), top=2)


# In[ ]:


def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
   
    input_shape = image_shape + (3,)
    #include_top=False - không muốn bao gồm các lớp fully connected (FC) ở cuối của mô hình -- 
    #sử dụng base_model để trích xuất đặc trưng và sau đó thêm lớp fully connected tùy chỉnh
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,include_top=False,weights='imagenet') 
    
    # Freeze the base model by making it non trainable
    base_model.trainable = False
    
    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    
    # apply data augmentation to the inputs
    x = data_augmentation(inputs)
    x = preprocess_input(x) 
    
    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False) 

    #Tổng hợp toàn cục
    x = tfl.GlobalAveragePooling2D()(x)
    #include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.2)(x)
        
    # create a prediction layer with one neuron (as a classifier only needs one)
    prediction_layer = tfl.Dense(1) 
    outputs = prediction_layer(x) 
    model = tf.keras.Model(inputs, outputs)
    
    return model


# In[ ]:


model2 = alpaca_model(IMG_SIZE, data_augmentation)


# In[ ]:


base_learning_rate = 0.001
model2.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


initial_epochs = 5
history = model2.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)


# In[ ]:




