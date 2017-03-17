import os
import csv
import numpy as np
import pandas as pd
import cv2
import math
import tensorflow as tf
tf.python.control_flow_ops = tf

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ELU, Dropout, Cropping2D, Conv2D, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import random_shift
from keras import backend as K
K.set_image_dim_ordering('tf')

import matplotlib.pyplot as plt



## Defining variables
pr_threshold = 1
new_size_col = 64
new_size_row = 64

def pd_read_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df2 = df[1:]
    return df



def add_brightness_to_image(img):
    # convert to HSV
    image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_factor = np.random.uniform() + 0.20
    # Apply the brightness reduction to the V channel
    image[:, :, 2] = image[:, :, 2] * random_factor
    image[:, :, 2][image[:, :, 2]>255] = 255 
    # convert to RBG again
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
            image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return image


def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)    
    #image = image/255.-.5
    return image



def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang,tr_x



def preprocess_image_file_train(line_data):
    # Preprocessing training files and augmenting
        image = np.zeros((new_size_row, new_size_col, 3))
        angle = np.zeros(1)

        camera_type = np.random.choice(['center', 'left', 'right'])

        if (camera_type == "center"):
            angle_adjust = 0.0
            col_index = 0
        if (camera_type == "left"):
            angle_adjust = 0.25
            col_index = 1
        if (camera_type == "right"):
            angle_adjust =  -0.25
            col_index = 2

        img_value =  line_data[col_index][0].strip()
        cwd = os.getcwd()
        name = cwd + '/data_udacity/IMG/' + img_value.split('/')[-1]
        # Add image and angle
        if os.path.isfile(name):
            angle = angle_adjust + float(line_data[3][0])
            image = cv2.imread(name)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image,angle,tr_x = trans_image(image,angle,150)
            image = augment_brightness_camera_images(image)
            image = add_random_shadow(image)
            image = preprocessImage(image)
            image = np.array(image)
            ind_flip = np.random.randint(2)
            if ind_flip==0:
                image = cv2.flip(image,1)
                angle = -angle
        else:
            print("orig",img_value)
            print(name)
            
        return image,angle



def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def generator(data, batch_size=32):
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)
    while 1:  # Loop forever so the generator never terminates
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()

            keep_pr = 0
            #x,y = preprocess_image_file_train(line_data)
            while keep_pr == 0:
                x,y = preprocess_image_file_train(line_data)
                pr_unif = np.random
                if abs(y)<.1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1

            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering





def nvidia_model(input_shape=(64, 64, 3)):
    dropout_prob = 0.5
    model = Sequential()

    def resize(image):
        import tensorflow as tf
        return tf.image.resize_images(image, (66, 200))

    # model.add(Cropping2D(((40, 20), (0, 0)),
    #                      input_shape=(160, 320, 3), name="Crop"))
    # model.add(Lambda(resize, name="Resize"))
    model.add(Lambda(lambda x: x / 255 -0.5,
                     input_shape=input_shape,
                     output_shape=input_shape, name="Normalize"))
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(
        2, 2), border_mode="valid", name="Conv2D_24"))
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(
        2, 2), border_mode="valid", name="Conv2D_36"))
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(
        2, 2), border_mode="valid", name="Conv2D_48"))
    model.add(Convolution2D(64, 3, 3, activation='elu', subsample=(
        1, 1), border_mode="valid", name="Conv2D_64_1"))
    model.add(Convolution2D(64, 3, 3, activation='elu', subsample=(
        1, 1), border_mode="valid", name="Conv2D_64_2"))
    model.add(Flatten(name="Flatten"))
    model.add(Dropout(dropout_prob, name="dropout"))
    model.add(Dense(100, activation='elu', name="dense_100"))
    model.add(Dense(50, activation='elu', name="dense_50"))
    model.add(Dense(1, activation='elu', name="dense_1"))
    return model


# Change the dir name where IMG and driving_log.csv are stored
#all_data_dirs=["data_uvs3","data_udacity","data_uvs_reverse","data_uvs5_half"]
all_data_dirs=[ "data_udacity"]
img_data =[]
img_angle = []
for data_dir in all_data_dirs:
    img_data_file   = data_dir + '/driving_log.csv'
    samples         = pd_read_data(img_data_file)
    #processed_data  = preprocess_data(samples, data_dir)
    #print("len of dir:", data_dir, " =", len(processed_data[0]))
    #img_data.extend(processed_data[0])
    #print("len after extend:", data_dir, " =", len(img_data))
    #img_angle.extend(processed_data[1])


image2=cv2.imread(os.getcwd()+'/data_udacity/IMG/'+samples.loc[3][0].split('/')[-1])

rows,cols,channels = image2.shape

#
# compile and train the model using the generator function
train_generator = generator(samples, batch_size=256)
validation_generator = generator(samples.sample(n=2000), batch_size=256)

# Build Model
model = nvidia_model()

# Display model summary
model.summary()

# Compile and Train the model
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
model.compile(loss='mse', optimizer=optimizer)

model.fit_generator(train_generator,
                    samples_per_epoch=len(samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(samples),
                    nb_epoch=15,
                    verbose=1)


# Save the model
model.save('model.h5')
with open("model.json", "w") as f:
    f.write(model.to_json())
