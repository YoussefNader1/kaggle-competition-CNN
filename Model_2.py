import os
from random import shuffle

from matplotlib import pyplot as plt
from tflearn import local_response_normalization

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

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 6, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)

if (os.path.exists('model_2.tfl.meta')):
    model.load('./model_2.tfl')
else:
    model.fit(X_train, y_train, n_epoch=10, show_metric=True, snapshot_step=100)
    model.save('model_2.tfl')

prediction = model.predict(X_test)
print(prediction[0])
