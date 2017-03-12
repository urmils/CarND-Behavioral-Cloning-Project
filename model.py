import os
import csv
import cv2
import sklearn
import numpy as np
import pandas as pd

import tensorflow as tf
tf.python.control_flow_ops = tf  

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ELU, Dropout, Cropping2D,Conv2D, AveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import random_shift
from keras import backend as K
K.set_image_dim_ordering('tf')

import matplotlib.pyplot as plt

# -------For Debug only ---------
def pd_read_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Pre process data (add flip)
# Add images from All 3 camera types
# Flip Images
# Add brightness
def preprocess_data2_working(samples,dir_name):
    images = []
    angles = []
    
    
    for batch_sample in samples:
        camera_type = np.random.choice(['center', 'left', 'right'])

        if (camera_type == "center"):
            angle_adjust = 0
            col_index = 0
        if (camera_type == "left"):
            angle_adjust = 0.2
            col_index = 1
        if (camera_type == "right"):
            angle_adjust = -0.2
            col_index = 2   
        
        name  = dir_name + '/IMG/'+batch_sample[col_index].split('/')[-1]
        angle = angle_adjust + round(float(batch_sample[3]))       
        
        # Add image and angle
        if os.path.isfile(name):
            image = cv2.imread(name)
        
        # This is done to reduce the bias for turning left that is present in the training data
        if abs(angle) > 0.30:
            angle = -1*angle
            image = cv2.flip(image, 1)
        
        # Apply random vertical shift 
        image = random_shift(image, 0, 0.2, 0, 1, 2)  # only vertical

        # Apply brightness to randomly selected image
        if np.random.random_sample() > 0.3:
            image = augment_brightness_camera_images(image)
        
        # Remove images where steering angle is close to 0
        if abs(angle) > 0.10:
            images.append(image)
            angles.append(angle)     

    return (images, angles)

# Pre process data (add flip)
# Data location: "data" folder contains IMG subfolder and driving_log.csv
# "data" and model.py are in same folder at same level
def preprocess_data_old(samples):
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

#---------------debug end----------------------



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

def add_brightness_to_image(img):
    # convert to HSV 
    image = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_factor = np.random.uniform() + 0.20
    # Apply the brightness reduction to the V channel
    image[:,:,2] = image[:,:,2]*random_factor
    # convert to RBG again
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image


def random_darken(image):
    w, h = image.size

    # Make a random box.
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(x1, w), random.randint(y1, h)

    # Loop through every pixel of our box (*GASP*) and darken.
    for i in range(x1, x2):
        for j in range(y1, y2):
            new_value = tuple([int(x * 0.5) for x in image.getpixel((i, j))])
            image.putpixel((i, j), new_value)
    return image


# Pre process data (add flip)
# Add images from All 3 camera types
# Flip Images
# Add brightness
def preprocess_data(samples,dir_name):
    images = []
    angles = []
    total_angles = []
    total_images = []
    

    for batch_sample in samples:
        camera_type = np.random.choice(['center', 'left', 'right'])

        if (camera_type == "center"):
            angle_adjust = 0
            col_index = 0
        if (camera_type == "left"):
            angle_adjust = 0.20
            col_index = 1
        if (camera_type == "right"):
            angle_adjust = -0.20
            col_index = 2   
        
        name  = dir_name + '/IMG/'+batch_sample[col_index].split('/')[-1]
        angle = angle_adjust + round(float(batch_sample[3]))       
        

        # Add image and angle
        if os.path.isfile(name):
            image = cv2.imread(name)
            images.append(image)  
            angles.append(angle)

   
    l1 = rnd_flip_images(images,angles)
    #l2 = rnd_select_nonzero_angles(images,angles)
    l3 = rnd_adjust_brightness_images(images,angles)
    l4 = rnd_shift_vertical_images(images,angles)

    total_images =   l1[0] + l3[0] + l4[0]
    total_angles =   l1[1] + l3[1] + l4[1]

    return (total_images, total_angles)


def rnd_flip_images(images,angles):
    
    total_images = []
    total_angles = []
    
    for index, item in enumerate(images):
        # Randomly flip images
        if np.random.random_sample() > 0.9:
            new_angle = -1*angles[index]
            new_image = cv2.flip(item, 1)
            total_images.append(new_image)
            total_angles.append(new_angle)
    return (total_images,total_angles)


def rnd_select_nonzero_angles(images,angles):
    
    total_images = []
    total_angles = []
    print("angles type=", type(angles))
    print("some angle values=", angles[0:5])
    for index, item in enumerate(images):
        # Randomly remove images where steering angle is near 0
        if np.random.random_sample() > 0.9:
            if abs(angles[index]) > 0.10:
                total_images.append(item)
                total_angles.append(angles[index])

    return (total_images,total_angles)

def remove_straight_angles(data):
    images = data[0]
    angles = data[1]
    smaller_dataset = rnd_select_nonzero_angles(images,angles)
    return smaller_dataset

def rnd_adjust_brightness_images(images,angles):
    
    total_images = []
    total_angles = []
    
    for index, item in enumerate(images):
        # Randomly add brightness to images
        if np.random.random_sample() > 0.4:
            image1 = add_brightness_to_image(item)
            total_images.append(image1)
            total_angles.append(angles[index])
    return (total_images,total_angles)

def rnd_shift_vertical_images(images,angles):
    
    total_images = []
    total_angles = []
    
    for index, item in enumerate(images):
        # Randomly vertical shift images
        if np.random.random_sample() > 0.6:
            image2 = random_shift(item, 0, 0.2, 0, 1, 2)  # only vertical
            total_images.append(image2)
            total_angles.append(angles[index])
    return (total_images,total_angles)




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

    model.add(Cropping2D(((1,1),(0,0)),input_shape=(66,200,3),name="Crop"))
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


# Change the dir name where IMG and driving_log.csv are stored
data_dir = 'data_udacity'
img_data_file = data_dir+'/driving_log.csv'
samples = read_img_file(img_data_file)
processed_data = preprocess_data(samples,data_dir)
print("len before removing angles=", len(processed_data[0]))
new_samples = remove_straight_angles(processed_data)


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
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(loss='mse', optimizer=optimizer)

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
