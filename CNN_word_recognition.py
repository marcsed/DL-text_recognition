# importing necessary libraries
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Idea
# we need to extract the word images from the word folder on my computer

# Notes
# need to get rid of any words with "er" in second column, it represents bad segmentation of word when extracting from paragraphs (see a01-000u-03-01 in dataset)

# architecture of words folder
# folder.words >
#   folder.words (76 items)>
#       folder.a01 - folder.r06  (15-30 items)>
#           folder.a01-000u - folder.r06-143 (50-100 words) >
#               img.a01-000u-00-00 - img.r06-143-04-10

# explaining the file naming convention
#  a01-000u-00-01
#       'a01-000u' represents a specific form
#       '00-01' represents the 2nd word on the 1st line of form a01-000u
#       '03-04' would represent the 5th word on the 4th line of form a01-000u


#######################################
#   loading in full image data set    #
#######################################
# setting up directory and initializing img data array list
DATADIR = "C:/Users/Administrator/Documents/full_word_folder"
img_data = []

# looping thru each img in folder and inputting them into an array of arrays
for img in os.listdir(DATADIR):
    try:
        img_array = cv2.imread(os.path.join(DATADIR, img))
        img_data.append(img_array)
    except Exception as e:
        pass

img_data = np.array(img_data)

###############################
#   loading in label data     #
###############################
#
os.chdir("C:/Users/Administrator/Documents")
label_data = pd.read_excel('words_fix.xlsx', header=None)
# extracting labels from data frame as array
labels = label_data[8].values


# checking that img and label data are same length
len(img_data)
len(labels)

####################################################
#   ensuring image and label data are matched up   #
####################################################

# indices confirmed: 500, 12211, 12345, 57000, 86543, 110000, 113620
labels[113620]
plt.imshow(img_data[113620], cmap=plt.cm.binary)
plt.show()

#######################################################
#   removing images with outlier pixel dimensions     #
#######################################################
# the index of the problem image is 113621, 4152
img_data[4152].shape
img_data[113621].shape

# remove images with outlier pixel dimensions from both img and label data
img_data = np.delete(img_data, [113621, 4152])
labels = np.delete(labels, [113621, 4152])

######################################
#   checking range of img sizes      #
######################################
img_height = []
img_width = []

for img in img_data:
    img_height.append(img_data[img].shape[0])
    img_width.append(img_data[img].shape[1])

img_height = np.array(img_height)
img_width = np.array(img_width)

# finding indices of images with outlier widths
result = np.where(img_width > 800)
outlier_indices = sorted(result[0], reverse=True)
len(outlier_indices)

# finding indices of images with outlier heights
result1 = np.where(img_height > 225)
outlier_indices1 = sorted(result1[0], reverse=True)
len(outlier_indices1)

full_outlier_indices = sorted(list(set(outlier_indices + outlier_indices1)), reverse=True)
len(full_outlier_indices)

# removing indices of images of sentences instead of words removed
img_data = np.delete(img_data, full_outlier_indices)
labels = np.delete(labels, full_outlier_indices)

# Making new img_height and width arrays
img_height = []
img_width = []

for img in img_data:
    img_height.append(img_data[img].shape[0])
    img_width.append(img_data[img].shape[1])

img_height = np.array(img_height)
img_width = np.array(img_width)

# Height Histogram
n_bins = 30

fig, axs = plt.subplots(1, 1,
                        figsize=(10, 7),
                        tight_layout=True)
axs.hist(img_height, bins=n_bins, edgecolor='black')
plt.xlabel("Image Height")
plt.ylabel("Count")
plt.title('Image Height Distribution')
plt.show()


# Width
n_bins = 30

fig, axs = plt.subplots(1, 1,
                        figsize=(10, 7),
                        tight_layout=True)
axs.hist(img_width, bins=n_bins, edgecolor='black')
plt.xlabel("Image Width")
plt.ylabel("Count")
plt.title('Image Width Distribution')
plt.show()


# indices reconfirmed: 500, 12211, 12345, 57000, 86543, 110000, 113620
labels[1235]
plt.imshow(img_data[1235], cmap=plt.cm.binary)
plt.show()

######################################
#   resizing images with padding     #
######################################
# Have to resize images to use fully connected layer for classification
img_data1 = img_data.tolist()

tf_images = tf.ragged.constant(img_data)
#tf_images = tf.ragged.constant(img_data1)

tf_img_data = tf.image.resize_with_crop_or_pad(tf_images, target_height=225, target_width=800)

#tf_images = tf.transpose(tf_images, perm=[0,2,1])
#tf_images = tf.expand_dims(tf_images, 2)
#resized = tf.image.resize_image_with_crop_or_pad(tf_images, height,width)



# setting up train and test splits
train_ratio = .85
train_index = int(len(img_data) * train_ratio)

# pair img/label and shuffle
X_train = img_data[:train_index]
X_test = img_data[train_index:]
y_train = labels[:train_index]
y_test = labels[train_index:]



# have to standardize array sizes before normalizing
#X_train = keras.utils.normalize(X_train)
#X_test = keras.utils.normalize(X_test)




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

