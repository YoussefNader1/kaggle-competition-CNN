import os
from random import shuffle

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import cv2
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

IMG_SIZE = 50
LR = 0.001
TRAIN_DIR = 'Train'
TEST_DIR = 'Test'


def create_label(image_name):
    """ Create one-hot encoded vector from image name
     Basketball -> 0
     Football -> 1
     Rowing -> 2
     Swimming -> 3
     Tennis -> 4
     Yoga -> 5
     """
    word_label = image_name.split('.')[0]
    # if "Basketball" in word_label
    if word_label.__contains__('Basketball'):
        return np.array([1, 0, 0, 0, 0, 0])
    elif word_label.__contains__('Football'):
        return np.array([0, 1, 0, 0, 0, 0])
    elif word_label.__contains__('Rowing'):
        return np.array([0, 0, 1, 0, 0, 0])
    elif word_label.__contains__('Swimming'):
        return np.array([0, 0, 0, 1, 0, 0])
    elif word_label.__contains__('Tennis'):
        return np.array([0, 0, 0, 0, 1, 0])
    elif word_label.__contains__('Yoga'):
        return np.array([0, 0, 0, 0, 0, 1])


def load_images_train_from_folder(folder):
    ''' Create a list of images for train and test

        folder:argument

    '''
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 1)
        img_normalized = img / 255.0
        img_data = cv2.resize(img_normalized, (IMG_SIZE, IMG_SIZE))
        if img is not None:
            images.append([np.array(img_data), create_label(filename)])
    shuffle(images)
    return images


def load_images_test_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 1)
        img_normalized = img / 255.0
        img_data = cv2.resize(img_normalized, (IMG_SIZE, IMG_SIZE))
        if img is not None:
            images.append([np.array(img_data)])
    return images


train = load_images_train_from_folder(TRAIN_DIR)
test = load_images_test_from_folder(TEST_DIR)

X_train = np.array([i[0] for i in train]).reshape((-1, IMG_SIZE, IMG_SIZE, 3))
y_train = [i[1] for i in train]

X_test = np.array([i for i in test]).reshape((-1, IMG_SIZE, IMG_SIZE, 3))

input_layer = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3])

block1_conv1 = conv_2d(input_layer, 64, 3, activation='relu', name='block1_conv1')
block1_conv2 = conv_2d(block1_conv1, 64, 3, activation='relu', name='block1_conv2')
block1_pool = max_pool_2d(block1_conv2, 2, strides=2, name='block1_pool')

block2_conv1 = conv_2d(block1_pool, 128, 3, activation='relu', name='block2_conv1')
block2_conv2 = conv_2d(block2_conv1, 128, 3, activation='relu', name='block2_conv2')
block2_pool = max_pool_2d(block2_conv2, 2, strides=2, name='block2_pool')

block3_conv1 = conv_2d(block2_pool, 256, 3, activation='relu', name='block3_conv1')
block3_conv2 = conv_2d(block3_conv1, 256, 3, activation='relu', name='block3_conv2')
block3_conv3 = conv_2d(block3_conv2, 256, 3, activation='relu', name='block3_conv3')
block3_conv4 = conv_2d(block3_conv3, 256, 3, activation='relu', name='block3_conv4')
block3_pool = max_pool_2d(block3_conv4, 2, strides=2, name='block3_pool')

block4_conv1 = conv_2d(block3_pool, 512, 3, activation='relu', name='block4_conv1')
block4_conv2 = conv_2d(block4_conv1, 512, 3, activation='relu', name='block4_conv2')
block4_conv3 = conv_2d(block4_conv2, 512, 3, activation='relu', name='block4_conv3')
block4_conv4 = conv_2d(block4_conv3, 512, 3, activation='relu', name='block4_conv4')
block4_pool = max_pool_2d(block4_conv4, 2, strides=2, name='block4_pool')

block5_conv1 = conv_2d(block4_pool, 512, 3, activation='relu', name='block5_conv1')
block5_conv2 = conv_2d(block5_conv1, 512, 3, activation='relu', name='block5_conv2')
block5_conv3 = conv_2d(block5_conv2, 512, 3, activation='relu', name='block5_conv3')
block5_conv4 = conv_2d(block5_conv3, 512, 3, activation='relu', name='block5_conv4')
block4_pool = max_pool_2d(block5_conv4, 2, strides=2, name='block4_pool')
flatten_layer = tflearn.layers.core.flatten(block4_pool, name='Flatten')

fc1 = fully_connected(flatten_layer, 4096, activation='relu')
dp1 = dropout(fc1, 0.5)
fc2 = fully_connected(dp1, 4096, activation='relu')
dp2 = dropout(fc2, 0.5)

network = fully_connected(dp2, 6, activation='softmax')

regression = tflearn.regression(network, optimizer='adam',
                                loss='categorical_crossentropy',
                                learning_rate=0.001)

model = tflearn.DNN(regression)

if (os.path.exists('model_2.tfl.meta')):
    model.load('./model_2.tfl')
else:
    model.fit(X_train, y_train, n_epoch=150, show_metric=True, snapshot_step=500)
    model.save('model_2.tfl')

prediction = model.predict(X_test)
print(prediction[0])
