import emnist
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

# load data
x_train, y_train = emnist.extract_training_samples('balanced')
x_test, y_test = emnist.extract_test_samples('balanced')

# change numeric character code to string characters (digits/letters)
def numeric_code_to_characters(label_list):
    for i in range(len(label_list)):
        if label_list[i] == 10:
            label_list[i] = "A"
        elif label_list[i] == 11:
            label_list[i] = "B"
        elif label_list[i] == 12:
            label_list[i] = "C"
        elif label_list[i] == 13:
            label_list[i] = "D"
        elif label_list[i] == 14:
            label_list[i] = "E"
        elif label_list[i] == 15:
            label_list[i] = "F"
        elif label_list[i] == 16:
            label_list[i] = "G"
        elif label_list[i] == 17:
            label_list[i] = "H"
        elif label_list[i] == 18:
            label_list[i] = "I"
        elif label_list[i] == 19:
            label_list[i] = "J"
        elif label_list[i] == 20:
            label_list[i] = "K"
        elif label_list[i] == 21:
            label_list[i] = "L"
        elif label_list[i] == 22:
            label_list[i] = "M"
        elif label_list[i] == 23:
            label_list[i] = "N"
        elif label_list[i] == 24:
            label_list[i] = "O"
        elif label_list[i] == 25:
            label_list[i] = "P"
        elif label_list[i] == 26:
            label_list[i] = "Q"
        elif label_list[i] == 27:
            label_list[i] = "R"
        elif label_list[i] == 28:
            label_list[i] = "S"
        elif label_list[i] == 29:
            label_list[i] = "T"
        elif label_list[i] == 30:
            label_list[i] = "U"
        elif label_list[i] == 31:
            label_list[i] = "V"
        elif label_list[i] == 32:
            label_list[i] = "W"
        elif label_list[i] == 33:
            label_list[i] = "X"
        elif label_list[i] == 34:
            label_list[i] = "Y"
        elif label_list[i] == 35:
            label_list[i] = "Z"
        elif label_list[i] == 36:
            label_list[i] = "a"
        elif label_list[i] == 37:
            label_list[i] = "b"
        elif label_list[i] == 38:
            label_list[i] = "d"
        elif label_list[i] == 39:
            label_list[i] = "e"
        elif label_list[i] == 40:
            label_list[i] = "f"
        elif label_list[i] == 41:
            label_list[i] = "g"
        elif label_list[i] == 42:
            label_list[i] = "h"
        elif label_list[i] == 43:
            label_list[i] = "n"
        elif label_list[i] == 44:
            label_list[i] = "q"
        elif label_list[i] == 45:
            label_list[i] = "r"
        elif label_list[i] == 46:
            label_list[i] = "t"

    return label_list

# get string character array for y_train and y_test
y_train_chars = numeric_code_to_characters(list(y_train))
y_test_chars = numeric_code_to_characters(list(y_test))

# visualize sample data
def display_sample(num, sample="Prediction"):
    if sample=="Train":
        # Print the label
        label = y_train_chars[num]
        # Reshape the 768 values to a 28x28 image
        image = x_train[num].reshape([28,28])
        # Plot image with sample, number and label
        plt.title('%s #%d – Label: %s' % (sample, num, label))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()
    else:
        # Print the label converted back to a number
        label = y_test_chars[num]
        # Reshape the 768 values to a 28x28 image
        image = x_test[num].reshape([28, 28])
        # Get string character prediction
        pred = predictions_chars[num]
        # Plot image with sample, number, true label and prediction
        fig = plt.figure()
        if label==pred:
            fig.patch.set_facecolor('xkcd:green')
        else:
            fig.patch.set_facecolor('xkcd:crimson')
        plt.title('%s #%d – Label: %s, Prediction: %s' % (sample, num, label, pred))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()

# test visualization for first train image
display_sample(0, "Train")

# scale images to 0-1 (dynamic for both 'channel_first' and 'channel_last'
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
# Another dropout layer
model.add(keras.layers.Dropout(0.5))
# Final categorization from 0-9,A-z with softmax
model.add(keras.layers.Dense(47, activation='softmax')) #47 for 47 different characters (digits and letters)
# Model description:
model.summary()


# fit model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.cnn = model.fit(x_train, y_train,
                      batch_size=32,
                      epochs=10,
                      verbose=2,
                      validation_data=(x_test, y_test))

# evaluate model on test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', round(score[0]*100,2))
print('Test accuracy:', round(score[1]*100,2))

# save model
model.save('CNN_digits_letters.model')
new_model = keras.models.load_model('CNN_digits_letters.model')

# make predictions
### nparray with probability vector for each observation
predictions_probs = new_model.predict(x_test)
### array with most-likely character (numeric code)
predictions = []
for i in range(len(predictions_probs)):
    predictions.append(np.argmax(predictions_probs[i]))
### array with string prediction
predictions_chars = numeric_code_to_characters(predictions)

# visualize/check predictions
### correct prediction
display_sample(3)
### false prediction
display_sample(4)