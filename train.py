from PIL import Image
import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

base_dir = ('D://vgg19/ac')
os.makedirs(base_dir, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.15,
    height_shift_range=0.15, shear_range=0.15,
    zoom_range=0.2,horizontal_flip=True,
    fill_mode="nearest", validation_split=0.3)
batch_size = 32

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')




conv_base19 = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))


def extract_features19(trainorval, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))
    labels = np.zeros(shape=(sample_count, 4))
    if trainorval=="training":
        generator = train_generator
    else:
        generator = val_generator
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base19.predict(preprocess_input(inputs_batch))
        try:
            features[i * batch_size : (i + 1) * batch_size] = features_batch
            labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        except ValueError:
            break
        if i==0:
            print("one down")
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features19, train_labels19 = extract_features19('training', 2351)
validation_features19, validation_labels19 = extract_features19('validation', 1004)
 
train_features19 = np.reshape(train_features19, (2351, 7 * 7 * 512))
validation_features19 = np.reshape(validation_features19, (1004, 7 * 7 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  #Removing 50% of the weights!
model.add(layers.Dense(4, activation='softmax'))


from tensorflow.keras.optimizers import Adam

INIT_LR = 1e-3
EPOCHS = 30
momentum = 0.9
nesterov = True

# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=momentum, nesterov=nesterov)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint


model_checkpoint = ModelCheckpoint(
    filepath='VGG19.h5',  # Filepath to save the best model
    monitor='val_acc',  # Monitor validation accuracy
    save_best_only=False,  # Save only the best model
    mode='max',  # The mode can be 'max' for accuracy, or 'min' for loss
    verbose=1  # Display a message when the best model is saved
)


history = model.fit(train_features19, train_labels19,
                    epochs=100,
                    batch_size=batch_size,
                    validation_data=(validation_features19, validation_labels19),
                    callbacks=[model_checkpoint])
