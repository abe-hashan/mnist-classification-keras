#!/usr/bin/env python
# coding: utf-8

# In[108]:


#Imports

from tensorflow import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import numpy as np


# In[109]:


# Load pre-shuffled MNIST data into train and test sets

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[110]:


#Exploring shape of test set

print(X_train.shape) 
print(y_train.shape) 
print(X_test.shape) 
print(y_test.shape) 


# In[111]:


#Plotting the first image of training set

plt.imshow(X_train[0])


# In[112]:


# reshaping the dimensions of images upto 4 as required by keras

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# See how it is rehaped

input_shape = (28, 28, 1)

# Making all the values as float to preserve decimal points after a division

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizing the values to be in the range of 0 - 1

X_train /= 255
X_test /= 255

print(X_train.shape) 
print(X_test.shape) 


# In[113]:


print(y_train)

# one hot encoding train and test classes 
class_count = 10
y_train = keras.utils.to_categorical(y_train, class_count)
y_test = keras.utils.to_categorical(y_test, class_count)


# In[79]:


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(class_count, activation="softmax"),
    ]
)

model.summary()


# In[81]:


batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[114]:


score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# In[ ]:




