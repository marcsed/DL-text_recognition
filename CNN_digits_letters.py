import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import emnist

# load data
# x_train, y_train = emnist.extract_training_samples('digits')
# x_test, y_test = emnist.extract_test_samples('digits')
# x_train, y_train = emnist.extract_training_samples('letters')
# x_test, y_test = emnist.extract_test_samples('letters')
x_train, y_train = emnist.extract_training_samples('balanced')
x_test, y_test = emnist.extract_test_samples('balanced')

# visualize sample data
def display_sample(num, sample):
    if sample=="train":
        #Print the one-hot array of this sample's label
        print(y_train[num])
        #Print the label converted back to a number
        label = y_train[num]
        #Reshape the 768 values to a 28x28 image
        image = x_train[num].reshape([28,28])
    else:
        # Print the one-hot array of this sample's label
        print(y_test[num])
        # Print the label converted back to a number
        label = y_test[num]
        # Reshape the 768 values to a 28x28 image
        image = x_test[num].reshape([28, 28])
    plt.title('Sample: %s, %d  Label: %d' % (sample, num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

display_sample(0, "train")

# scale images
if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255



# build model – Convolutional NNet
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=input_shape))
# 64 3x3 kernels
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
# Reduce by taking the max of each 2x2 block
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout to avoid overfitting
model.add(keras.layers.Dropout(0.25))
# Flatten the results to one dimension for passing into our final layer
model.add(keras.layers.Flatten())
# A hidden layer to learn with
model.add(keras.layers.Dense(128, activation='relu'))
# Another dropout
model.add(keras.layers.Dropout(0.5))
# Final categorization from 0-9,A-z with softmax
model.add(keras.layers.Dense(47, activation='softmax'))
# Let's double check the model description:
model.summary()


# fit model
model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

# train model
model.cnn = model.fit(x_train, y_train,
                      batch_size=32,
                      epochs=1, #10 – for better accuracy
                      verbose=2,
                      validation_data=(x_test, y_test))

# evaluate model on test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model
model.save('CNN_digits_letters.model')
new_model = keras.models.load_model('CNN_digits_letters.model')

# make predictions
predictions = new_model.predict(x_test)

# visualize/check predictions
print(np.argmax(predictions[1]))
display_sample(1, "test")