import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

#image directory for reading
image_directory = 'D:\\vgg19\\project\\dataset\\'


# get the list of images for each rice type
BrownSpot_images = os.listdir(image_directory + 'BrownSpot/')
Healthy_images = os.listdir(image_directory + 'Healthy/')
Hispa_images = os.listdir(image_directory + 'Hispa/')
LeafBlast_images = os.listdir(image_directory + 'LeafBlast/')

dataset = []
label = []

# print(len(MildDemented_images))
# print(len(ModerateDemented_images))
# print(len(NonDemented_images))
# print(len(VeryMildDemented_images))


# set the input size for the images
INPUT_SIZE = 64

# loop through the images for MildDemented
for i, image_name in enumerate(BrownSpot_images):
    if(image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'BrownSpot/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append([0])    #{1,0,0,0}

# loop through the images for Healthy 
for i, image_name in enumerate(Healthy_images):
    if(image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Healthy/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append([1])    ##{0,1,0,0}

# loop through the images for Hispa
for i, image_name in enumerate(Hispa_images):
    if(image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Hispa/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append([2]) ##{0,0,1,0}

# loop through the images for LeafBlast
for i, image_name in enumerate(LeafBlast_images):
    if(image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'LeafBlast/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append([3]) ##{0,0,0,1}

#print(ModerateDemented_images)
#print()

# convert the dataset and label lists to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# normalize the input data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# convert the labels to categorical format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# build the CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))


#below code is VGG16 algorithm

# model = models.Sequential()
# model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.2))  #Removing 50% of the weights!
# model.add(layers.Dense(4, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# training the model
#model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), )


model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))


model.save('VGG19.h5')