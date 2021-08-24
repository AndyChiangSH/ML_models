# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:31:19 2021

@author: andy1
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

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
    model.add(Dense(units=512, input_dim=28*28, kernel_initializer="normal", activation="relu"))
    # output layer
    model.add(Dense(units=10, kernel_initializer="normal", activation="softmax"))
    # setting model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # train model
    
    try:
        model.load_weights("mnist_demo_model2_weight.h5")
        print("load weight complete!")
    except:
        print("Can't load weight, start training new model!")
        
    history = model.fit(x=train_feature_vector, y=train_label_onehot, validation_split=0.2, epochs=1, batch_size=200, verbose=1)
    
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
    
    # save model and weight
    model.save_weights("mnist_demo_model2_weight.h5")
    print("save weight complete!")
    model.save("mnist_demo_model2.h5")
    print("save model complete!")
    del model
    