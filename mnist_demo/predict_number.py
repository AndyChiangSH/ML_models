# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:38:00 2021

@author: andy1
"""

import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

def show_images_labels(images, labels, predictions, start_id=10, num=10):
    plt.gcf().set_size_inches(12, 14)
    if num > 25:
        num = 25
        
    ac = 0
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
                ac += 1
            else:
                title += " (x)"
            title += f"\nlabel = {labels[start_id]}"
            
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id += 1
     
    print("acc =", ac)
    acc_rate = ac/float(len(predictions)) * 100
    print(f"acc rate = {acc_rate}%")
    plt.show()

if __name__ == '__main__':
    
    test_feature = []
    test_label = []
    
    files = glob.glob(r"mydata\2\*.jpg")
    print("files =", len(files))
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 圖片黑白二極化
        img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)[1]
        # 圖片黑白反轉
        # img = (255-img)/255
        # print(img)
        # plt.imshow(img, cmap="binary")
        # plt.show()

        test_feature.append(img)
        test_label.append(int(file[9]))
        
    test_feature = np.array(test_feature)
    test_label = np.array(test_label)
    
    test_feature_vector = test_feature.reshape(len(test_feature), 784).astype("float32")
    
    # load model
    model = load_model("mnist_demo_model_dropout.h5")
    print("load model complete!")
    
    # predict testing data
    predict_test = model.predict(test_feature_vector)
    prediction = np.argmax(predict_test, axis=1)
    
    show_images_labels(test_feature, test_label, prediction, 0, len(test_feature))
    print("predice done")
    
    
"""
1、6、9 最常出事
5、8 偶爾

784 -> 256 -> 10：70%
784 -> 256 -> 256 -> 10：80%
784 -> 256 -> 256 -> 256 -> 10：80%
784 -> 256 -> 256 -> 256 -> 256 -> 10：70%
784 -> 256 -> 256 -> 256 -> 256 -> 256 -> 10：80%(!?)

784 -> 256 -> 10：70%
784 -> 512 -> 10：80%
784 -> 1024 -> 10：80%

784 -> 512 -> 512 -> 10：90%

784 -> 512 -> 10 (D：0.2)：80%
784 -> 512 -> 10 (D：0.5)：70%
"""