# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:46:51 2021

@author: andy1
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.layers import Dropout


def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap="binary")
    plt.show()


def show_images_labels(images, labels, predictions, start_id=10, num=10):
    plt.gcf().set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(num):
        # 指定格子位置
        ax = plt.subplot(5, 5, i+1)
        # 顯示黑白圖片
        ax.imshow(images[start_id], cmap="binary")

        if len(predictions) == 0:   # 沒有預測資料，只顯示label
            title = f"label = {labels[start_id]}"
        else:   # 有預測資料，顯示預測結果
            title = f"predict = {predictions[start_id]}"
            # 預測正確印(o)，不正確印(x)
            if labels[start_id] == predictions[start_id]:
                title += " (o)"
            else:
                title += " (x)"
            title += f"\nlabel = {labels[start_id]}"

        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id += 1

    plt.show()


def feature_process(feature):
    return feature.reshape(len(feature), 784).astype("float32")/255


def label_process(label):
    return np_utils.to_categorical(label)


if __name__ == '__main__':

    (train_feature, train_label), (test_feature, test_label) = mnist.load_data()

    train_feature_vector = feature_process(train_feature)
    test_feature_vector = feature_process(test_feature)

    train_label_onehot = label_process(train_label)
    test_label_onehot = label_process(test_label)

    # create model
    model = Sequential()
    # input layer and hidden layer
    model.add(Dense(units=1024, input_dim=28*28,
                    kernel_initializer="normal", activation="relu"))
    model.add(Dropout(0.2))
    # output layer
    model.add(Dense(units=10, kernel_initializer="normal", activation="softmax"))

    print(model.summary())

    # setting model
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    # train model
    history = model.fit(x=train_feature_vector, y=train_label_onehot,
                        validation_split=0.2, epochs=20, batch_size=200, verbose=1)

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
    print(f"test lose: {scores[0]}")
    print(f"test acc: {scores[1]}")

    # predict testing data
    predict_test = model.predict(test_feature_vector)
    prediction = np.argmax(predict_test, axis=1)

    show_images_labels(test_feature, test_label, prediction, 0, 10)

    # save model
    model.save("mnist_demo_model_dropout.h5")
    print("save model complete!")
    del model

"""
784 -> 256(D：0.2) -> 10：0.9786
784 -> 512(D：0.2) -> 10：0.9794
784 -> 1024(D：0.2) -> 10：0.9831

784 -> 256(D：0.5) -> 10：0.9774
784 -> 512(D：0.5) -> 10：0.9792
784 -> 1024(D：0.5) -> 10：0.9821

784 -> 128(D：0.2) -> 128(D：0.2) -> 10：0.9776
784 -> 256(D：0.2) -> 256(D：0.2) -> 10：0.9820
784 -> 512(D：0.2) -> 512(D：0.2) -> 10：0.9803
784 -> 1024(D：0.2) -> 1024(D：0.2) -> 10：0.9791

784 -> 256(D：0.2) -> 256(D：0.2) -> 256(D：0.2) -> 10：0.9790
"""
