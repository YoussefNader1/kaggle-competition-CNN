import os
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


def load_images_from_folder(folder):
    ''' Create a list of images for train and test

        folder:argument

    '''
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        img_data = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if img is not None:
            images.append([np.array(img_data), create_label(filename)])
    return images


# conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# conv1 = conv_2d(conv_input, 32, 5, activation='relu')
# pool1 = max_pool_2d(conv1, 5)
#
# conv2 = conv_2d(pool1, 64, 5, activation='relu')
# pool2 = max_pool_2d(conv2, 5)
#
# conv3 = conv_2d(pool2, 128, 5, activation='relu')
# pool3 = max_pool_2d(conv3, 5)
#
# conv4 = conv_2d(pool3, 64, 5, activation='relu')
# pool4 = max_pool_2d(conv4, 5)
#
# conv5 = conv_2d(pool4, 32, 5, activation='relu')
# pool5 = max_pool_2d(conv5, 5)
#
# fully_layer = fully_connected(pool5, 1024, activation='relu')
# fully_layer = dropout(fully_layer, 0.5)
#
# cnn_layers = fully_connected(fully_layer, 2, activation='softmax')

train = load_images_from_folder(TRAIN_DIR)
test = load_images_from_folder(TEST_DIR)

print(train[0][1])
print("debug")