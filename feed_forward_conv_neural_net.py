#Simple fully connected neural network. Comparing fully connected vanilla neural net vs convolutional model for mnist and cifar datasets. Experiment with batch normalization.


import keras
import matplotlib.pyplot as plt
import imageio
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
import numpy as np

#Setup parameters
num_classes = 10
batch_size = 64
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols)
epochs = 4

#Import mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


print(y_train[0])
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Simple fully connected neural network with 2 hidden layers
model = Sequential()

model.add(Dense(units=200, activation='relu',input_shape = input_shape))
model.add(Flatten())
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

#Learning process and compile
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#Learning
model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size)


#Evaluation
score = model.evaluate(x_test,y_test, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#Compare against convolutional model. 

model1 = Sequential()
model1.add(Conv2D(32, (3,3),padding='same', strides = (2,2), activation='relu',input_shape = (28,28,1)))
model1.add(MaxPooling2D(pool_size = (2,2)))
model1.add(Conv2D(64, (3,3),padding='same', strides = (2,2), activation='relu'))
model1.add(MaxPooling2D(pool_size = (2,2)))
model1.add(Flatten())
model1.add(Dense(units=200, activation='relu'))
model1.add(Dense(units=200, activation='relu'))
model1.add(Dense(units=10, activation='softmax'))
model1.summary()

#Compile
model1.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#Learning
model1.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size)

#Evaluation
score1 = model1.evaluate(x_test,y_test, verbose = 1)
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])

#CIFAR10

from keras.datasets import cifar10
from keras.layers.normalization import BatchNormalization

#Import data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Fit data and set parameters
x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)
epochs = 25

#Convolutional model with batch normalization. 
model2 = Sequential()
model2.add(Conv2D(32, (3,3),padding='same', strides = (1,1), activation='relu',input_shape = (32,32,3)))
model2.add(BatchNormalization())
model2.add(Conv2D(32, (3,3),padding='same', strides = (1,1), activation='relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size = (2,2)))
model2.add(Dropout(0.2))

model2.add(Conv2D(64, (3,3),padding='same', strides = (1,1), activation='relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(64, (3,3),padding='same', strides = (1,1), activation='relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size = (2,2)))
model2.add(Dropout(0.3))


model2.add(Conv2D(128, (3,3),padding='same', strides = (1,1), activation='relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(128, (3,3),padding='same', strides = (1,1), activation='relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size = (2,2)))
model2.add(Dropout(0.4))

model2.add(Flatten())
model2.add(Dense(units=10, activation='softmax'))

model2.summary()

#Set lr and loss function. Compile.
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#Train model
model2.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size)

#Evaluation 
score2 = model2.evaluate(x_test,y_test, verbose = 1)
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])