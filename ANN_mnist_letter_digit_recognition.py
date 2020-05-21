import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from emnist import extract_training_samples, extract_test_samples

# The NNet has a validation loss rate of 54.11% while having a validation accuracy of 83.45%


# Loading in train/test data from emnist database
X_train, y_train = extract_training_samples('balanced')
X_test, y_test = extract_test_samples('balanced')


# Pre-processing Data before building model
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

if tf.keras.backend.image_data_format() == 'channels_first':
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

# simple feed forward neural network
model = Sequential()

# for simpler models, we want to image to be flattened, not 28x28 (for CNNs this is not true)
# you can use numpy to do this, or just use built in functionality of keras
model.add(Flatten())

# going with 2 hidden layers with 128 neurons each
# using "relu" activation function, which is typically the default for simpler models
model.add(Dense(128,
                activation=tf.nn.relu))
model.add(Dense(128,
                activation=tf.nn.relu))

# Add 1 output layer with 10 nuerons (for 10 possible digits)
# using a softmax activation function (b/c this is a probability distribution problem)
model.add(Dense(47,
                activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#training the model using 3 epochs
model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_test, y_test))


val_loss, val_accuracy = model.evaluate(X_test, y_test)
print("The NNet has a validation loss rate of " + str(round(val_loss*100, 2)) + "%"
      + "\n" + " while having a validation accuracy of " + str(round(val_accuracy*100, 2)) + "%")
