# importing necessary libraries
import tensorflow.keras as keras
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

# loading in full image data set

# setting up directory and initializing img data array list
DATADIR = "C:/Users/Administrator/Downloads/mega_word_folder/mega_word_folder"
img_data = []

# looping thru each img in folder and inputting them into an array of arrays
for img in os.listdir(DATADIR):
    try:
        img_array = cv2.imread(os.path.join(DATADIR, img))
        img_data.append(img_array)
    except Exception as e:
        pass

img_data = np.array(img_data)

# loading in label data
# had to adapt original text file to get rid of extra columns in some rows, caused by pictures with spaces in them
# have to use skiprows=18 if you don't edit txt file and delimitter=" "
os.chdir("C:/Users/Administrator/Documents")
label_data = pd.read_excel('words_fix.xlsx', header=None)
label_data.shape
label_data.head()

# lengths are 18 off because there are 18 lines at the top of the text file explaining the values, but not read in
# jk fixed now
len(img_data)
len(label_data)

# extracting labels from data frame as array
# labels are 1 value longer, is this because of the headers from dataframe?
labels = label_data[8].values
len(labels)

# setting up train and test splits
train_ratio = .85
train_index = int(len(img_data) * train_ratio)

#pair img/label and shuffle
X_train = img_data[:train_index]
X_test = img_data[train_index:]
y_train = labels[:train_index]
y_test = labels[train_index:]

#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')


#have to standardize array sizes before normalizing
X_train = keras.utils.normalize(X_train)
X_test = keras.utils.normalize(X_test)


plt.imshow(X_train[500], cmap=plt.cm.binary)
plt.show()
X_train[500].shape
