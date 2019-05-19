from keras.models import Sequential
import time
from keras.layers import Conv2D, Input
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.optimizers import *
from keras import backend as K
from keras.regularizers import l2

K.set_image_data_format('channels_last')
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
from sklearn.utils import shuffle
import glob

data_path = os.path.join('data/')


def init_weights(shape):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def init_bias(shape):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_siamese_model(shape):
    left_input = Input(shape)
    right_input = Input(shape)
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=shape,
                     kernel_initializer=init_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=init_weights,
                     bias_initializer=init_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=init_weights,
                     bias_initializer=init_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=init_weights,
                     bias_initializer=init_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=init_weights, bias_initializer=init_bias))
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid', bias_initializer=init_bias)(L1_distance)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net


class Siamese_Network:

    def __init__(self):

        file_path = r'data\train_resized'
        self.folder_train = (glob.glob(file_path + "\*"))
        self.d_folder_train = []
        for i in range(len(self.folder_train)):
            self.d_folder_train.append(glob.glob(self.folder_train[i] + "\*.bmp"))

        file_path2 = r"data\val_resized"
        self.folder_val = (glob.glob(file_path2 + "\*"))
        self.d_folder_val = []
        for i in range(len(self.folder_val)):
            self.d_folder_val.append(glob.glob(self.folder_val[i] + "\*.bmp"))

    def batch(self, batch_size):

        folders, pictures_in_folder = 26, 51

        categories = rng.choice(folders, size=(batch_size,), replace=True)
        pair = [np.zeros((batch_size, 200, 200, 3)) for i in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            cat = categories[i]
            pic_1 = rng.randint(0, pictures_in_folder)
            img = cv2.imread(self.d_folder_train[cat][pic_1])
            pair[0][i, :, :, :] = img
            pic_2 = rng.randint(0, pictures_in_folder)
            if i >= batch_size // 2:
                cat_2 = cat
            else:
                cat_2 = (cat + rng.randint(1, folders)) % folders
            img2 = cv2.imread(self.d_folder_train[cat_2][pic_2])

            pair[1][i, :, :, :] = img2
            return pair, targets

    def gen(self, batch_size):

        while True:
            pair, targets = self.batch(batch_size)
            yield (pair, targets)

    def prepare_pairs(self, N):

        folders = 26
        pictures_in_folder = 51

        indices = rng.randint(0, pictures_in_folder, size=(N,))
        test_image = [np.zeros((200, 200, 3)) for i in range(N)]
        support_set = [np.zeros((200, 200, 3)) for i in range(N)]
        categories = rng.choice(range(folders), size=(N,), replace=True)
        true_cat = categories[0]
        pic1, pic2 = rng.choice(pictures_in_folder, replace=False, size=(2,))
        img3 = cv2.imread(self.d_folder_train[true_cat][pic1])
        Image3 = img3.reshape(200, 200, 3)
        img4 = cv2.imread(self.d_folder_train[true_cat][pic2])

        for i in range(N):
            test_image[i][:, :, :] = Image3
            support_set[i][:, :, :] = cv2.imread(self.d_folder_train[categories[i]][indices[i]])

        support_set[0][:, :, :] = img4
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)

        pair = [test_image, support_set]

        return pair, targets

    def try_pairs(self, model, N, k):

        correct = 0

        for i in range(k):
            inputs, targets = self.prepare_pairs(N)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                correct += 1
        percent = (100.0 * correct / k)
        print("Training set")
        print("average {}% \n".format(percent))
        return percent

    def prepare_pairs_val(self, N):

        folders = 5
        pictures_in_folder = 51

        indices = rng.randint(0, pictures_in_folder, size=(N,))
        test_image = [np.zeros((200, 200, 3)) for i in range(N)]
        support_set = [np.zeros((200, 200, 3)) for i in range(N)]
        categories = rng.choice(range(folders), size=(N,), replace=True)
        true_cat = categories[0]
        pic1, pic2 = rng.choice(pictures_in_folder, replace=False, size=(2,))
        img3 = cv2.imread(self.d_folder_val[true_cat][pic1])
        Image3 = img3.reshape(200, 200, 3)
        img4 = cv2.imread(self.d_folder_val[true_cat][pic2])

        for i in range(N):
            test_image[i][:, :, :] = Image3
            support_set[i][:, :, :] = cv2.imread(self.d_folder_val[categories[i]][indices[i]])

        support_set[0][:, :, :] = img4
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)

        pair = [test_image, support_set]

        return pair, targets

    def try_pairs_val(self, model, N, k):

        correct = 0

        for i in range(k):
            inputs, targets = self.prepare_pairs_val(N)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                correct += 1
        percent = (100.0 * correct / k)
        print("Validation set")
        print("average {}% \n".format(percent))
        return percent


    def train(self, model):
        model.fit_generator(self.gen(batch_size))

def modeling():
    weights = init_weights((1000, 1))
    bias = init_bias((1000, 1))
    model = get_siamese_model((200, 200, 3))
    model.summary()

    optimizer = Adam(lr=0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    loader = Siamese_Network()

    milestone = 2
    batch_size = 32
    iterations = 100
    pics_at_once = 25
    num_of_pics = 500
    best = -1


    weights_path = os.path.join(data_path, "model.h5")
    model.load_weights(weights_path)
    return model

def training():

    print("Training has just started")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, iterations):
        (inputs, targets) = loader.batch(batch_size)
        loss = model.train_on_batch(inputs, targets)
        print("\n \n")
        print("Loss: {0} \n".format(loss))
        print("Iteration: {0}".format(i))
        if i % milestone == 0:
            print("Time for {0} iterations: {1}".format(i, time.time() - t_start))
            val_acc = loader.try_pairs(model, pics_at_once, num_of_pics)
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                weights_path = os.path.join(data_path, "model.h5")
                print("Saving weights \n")
                model.save_weights(weights_path)
                best = val_acc






def checking():
    cas = np.arange(1, 25, 1)
    val, train = [], []
    num_of_pics = 200
    for N in cas:
        val.append(loader.try_pairs_val(model, N, num_of_pics))
        train.append(loader.try_pairs(model, N, num_of_pics))

    fig, ax = plt.subplots(1)

    ax.plot(cas, train, label="training")
    ax.plot(cas, val, label="validation")
    plt.xlabel("Number of classes")
    plt.ylabel("% Accuracy")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
