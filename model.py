import os
import csv
import cv2
import sklearn
import numpy as np
import pandas as pd

import tensorflow as tf
tf.python.control_flow_ops = tf  

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ELU, Dropout, Cropping2D,Conv2D, AveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')


# Read data from data set
def read_img_file(file_path):
    samples = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    # Remove the column header from the list
    del samples[0]
    return samples

# Pre process data (add flip)
# Data location: "data" folder contains IMG subfolder and driving_log.csv
# "data" and model.py are in same folder at same level
def preprocess_data(samples):
    images = []
    angles = []
    tot_imgs = []
    tot_angles = []
    for batch_sample in samples:
        name = 'data/IMG/'+batch_sample[0].split('/')[-1]
        if os.path.isfile(name):
            center_image = cv2.imread(name)
            center_angle = round(float(batch_sample[3]))
            #images.append(center_image)
            #angles.append(center_angle)
            tot_imgs.append(center_image)
            tot_angles.append(center_angle)
            # Add flip images 
            if abs(center_angle) > 0.33:
                flip_center = cv2.flip(center_image, 1)
                center_angle *= -1
                tot_imgs.append(flip_center)
                tot_angles.append(center_angle)
        else:
            print('\nfile not valid', name)
    return (tot_imgs, tot_angles)
        
# yield data
def generator_wo_preprocess(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            tot_imgs = []
            tot_angles = []
            
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                if os.path.isfile(name):
                    center_image = cv2.imread(name)
                    center_angle = round(float(batch_sample[3]))
                    #images.append(center_image)
                    #angles.append(center_angle)
                    tot_imgs.append(center_image)
                    tot_angles.append(center_angle)
                    # Add flip images 
                    flip_center = cv2.flip(center_image, 1)
                    center_angle *= -1
                    tot_imgs.append(flip_center)
                    tot_angles.append(center_angle)
                else:
                    print('\nfile not valid', name)

            images = tot_imgs[offset:offset+batch_size]
            angles = tot_angles[offset:offset+batch_size]   
            # tensorflow expects data in float32 format
            X_train = np.array(images,dtype='float32')
            y_train = np.array(angles,dtype='float32')
            yield sklearn.utils.shuffle(X_train, y_train)

# yield data
def generator(preprocessed_data, batch_size=32):
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, len(preprocessed_data[0]), batch_size):
            images = preprocessed_data[0][offset:offset+batch_size]
            angles = preprocessed_data[1][offset:offset+batch_size]

        # tensorflow expects data in float32 format
        X_train = np.array(images,dtype='float32')
        y_train = np.array(angles,dtype='float32')
        yield sklearn.utils.shuffle(X_train, y_train)


def nvidia_model(input_shape=(66, 200, 3)):
    dropout_prob = 0.5 
    model = Sequential()

    def resize(image):
        import tensorflow as tf
        return tf.image.resize_images(image, (66, 200))       

    model.add(Cropping2D(((60,20),(1,1)),input_shape=(160,320,3),name="Crop"))
    model.add(Lambda(resize,name="Resize"))
    model.add(Lambda(lambda x: x / 255 - 0.5,
                     input_shape=input_shape,
                     output_shape=input_shape, name="Normalize"))
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2), border_mode="valid",name="Conv2D_24"))
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2), border_mode="valid",name="Conv2D_36"))
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2), border_mode="valid",name="Conv2D_48"))
    model.add(Convolution2D(64, 3, 3, activation='elu', subsample=(1, 1), border_mode="valid",name="Conv2D_64_1"))
    model.add(Convolution2D(64, 3, 3, activation='elu', subsample=(1, 1), border_mode="valid",name="Conv2D_64_2"))
    model.add(Flatten(name="Flatten"))
    model.add(Dropout(dropout_prob,name="dropout"))
    model.add(Dense(100, activation='elu',name="dense_100"))
    model.add(Dense(50, activation='elu',name="dense_50"))
    model.add(Dense(1, activation='elu',name="dense_1"))
    return model


# Start the program
img_data_file = 'data/driving_log.csv'

samples = read_img_file(img_data_file)
new_samples = preprocess_data(samples)

# Split Original data into Training and Validation (80/20)
X_train_samples, X_validation_samples = train_test_split(new_samples[0], test_size=0.2)

y_train_samples, y_validation_samples = train_test_split(new_samples[1], test_size=0.2)

# Display length of all dataset
print(len(new_samples[0]))
print(len(X_train_samples))
print(len(X_validation_samples))

train_samples = (X_train_samples, y_train_samples)
validation_samples = (X_validation_samples, y_validation_samples)

# compile and train the model using the generator function
train_generator      = generator(train_samples, batch_size=100)
validation_generator = generator(validation_samples, batch_size=100)

# Build Model
model = nvidia_model()

# Display model summary
model.summary()

# Compile and Train the model
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, 
                    samples_per_epoch = len(train_samples[0]), 
                    validation_data   = validation_generator, 
                    nb_val_samples    = len(validation_samples[0]),
                    nb_epoch          = 3,  
                    verbose           = 1)


# Save the model
model.save('model.h5')
with open("model.json", "w") as f:
    f.write(model.to_json())
