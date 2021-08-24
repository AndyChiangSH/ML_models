# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:43:07 2021

@author: andy1
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from keras.utils import np_utils

    
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
    
    test_feature_vector = feature_process(test_feature)
    test_label_onehot = label_process(test_label)
    
    # load model
    model = load_model("mnist_demo_model2.h5")
    print("load model complete!")
    
    # evaluate testing data
    scores = model.evaluate(test_feature_vector, test_label_onehot)
    print(f"test lose: {scores[0]}")
    print(f"test acc: {scores[1]}")
    
    # predict testing data
    predict_test = model.predict(test_feature_vector) 
    prediction = np.argmax(predict_test, axis=1)
    
    show_images_labels(test_feature, test_label, prediction, 10, 10)