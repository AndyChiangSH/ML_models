# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:31:19 2021

@author: andy1
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Conv2D
import numpy as np


def feature_process(feature):
    return feature.reshape(feature.shape[0], 28, 28, 1).astype("float32")/255


def label_process(label):
    return np_utils.to_categorical(label)


if __name__ == '__main__':

    (train_feature, train_label), (test_feature, test_label) = mnist.load_data()

    train_feature_vector = feature_process(train_feature)
    test_feature_vector = feature_process(test_feature)

    train_label_onehot = label_process(train_label)
    test_label_onehot = label_process(test_label)

    # create CNN model
    model = Sequential()
    # Convolution + Max Pooling
    model.add(Conv2D(filters=15, kernel_size=(
        3, 3), input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=30, kernel_size=(3, 3)))
    model.add(MaxPooling2D((2, 2)))
    # Flatten + DNN
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.summary()
    # setting model
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    # try:
    #     model.load_weights("mnist_demo_model3_weight.h5")
    #     print("load weight complete!")
    # except:
    #     print("Can't load weight, start training new model!")

    # train model
    history = model.fit(x=train_feature_vector, y=train_label_onehot,
                        validation_split=0.2, epochs=15, batch_size=100)

    # paint training loss graph
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title('Training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.show()

    # paint training accuracy graph
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title('Training accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.show()

    # evaluate testing data
    scores = model.evaluate(test_feature_vector, test_label_onehot)
    print(f"CNN test lose: {scores[0]}")
    print(f"CNN test acc: {scores[1]}")

    # save model and weight
    model.save_weights("mnist_demo_model3_weight.h5")
    print("save weight complete!")
    model.save("mnist_demo_model3.h5")
    print("save model complete!")
    del model
