import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import emnist

# The CNN has a loss rate of 38.68% while having an accuracy of 87.02%

# Loading in train/test data from emnist database
X_train, y_train = emnist.extract_training_samples('balanced')
X_test, y_test = emnist.extract_test_samples('balanced')


# Pre-processing Data before building model
if keras.backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


# Building a CNN
model = keras.models.Sequential()

# adding 2 convolutional layers with 64 neurons, relu activation functions, and .25 dropout
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

# flattening before dense layer (1d input for dense, 2d for conv layer
model.add(keras.layers.Flatten())

# going with 1 dense 128 neuron layer
model.add(keras.layers.Dense(64, activation='relu'))


# output layer with 47 neurons for the 47 different classes
model.add(keras.layers.Dense(47, activation='softmax'))

# need to define parameters for training of the model
model.compile(optimizer='adam',
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# training the model using 3 epochs
model.fit(X_train, y_train,
          batch_size=32,
          epochs=7,
          verbose=2,
          validation_data=(X_test, y_test))


# calculating loss and accuracy rates and outputing them in print statement
val_loss, val_accuracy = model.evaluate(X_test, y_test)
print("The NNet has a loss rate of " + str(round(val_loss*100, 2)) + "%"
      + "\n" + " while having an accuracy of " + str(round(val_accuracy*100, 2)) + "%")
