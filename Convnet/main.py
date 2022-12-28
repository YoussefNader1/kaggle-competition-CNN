import os
from random import shuffle
import csv
import numpy as np
import cv2
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = 50
TRAIN_DIR = r'\Train'
TEST_DIR = r'\Test'


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
    images_names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 1)
        img_normalized = img / 255.0
        img_data = cv2.resize(img_normalized, (IMG_SIZE, IMG_SIZE))
        if img is not None:
            images.append([np.array(img_data)])
            images_names.append(filename)
    return images, images_names


train = load_images_train_from_folder(TRAIN_DIR)
test, images_names = load_images_test_from_folder(TEST_DIR)

X_train = np.array([i[0] for i in train]).reshape((-1, IMG_SIZE, IMG_SIZE, 3))
y_train = [i[1] for i in train]

X_test = np.array([i for i in test]).reshape((-1, IMG_SIZE, IMG_SIZE, 3))

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 6, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.00005)

model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit(X_train, y_train, n_epoch=150, show_metric=True, validation_set=0.2)
    model.save('model.tfl')

predictions = model.predict(X_test)

pred = []
for prediction in predictions:
    max_val = np.argmax(prediction)
    pred.append(max_val)
print(pred)

headers = ["image_name", "label"]
OutPut_list = []
for i in range(len(pred)):
    x = [images_names[i], pred[i]]
    OutPut_list.append(x)
with open("Sports_test.csv", "w") as Sport:
    student = csv.writer(Sport)
    student.writerow(headers)
    student.writerows(OutPut_list)
