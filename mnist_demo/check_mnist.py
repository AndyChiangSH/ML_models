# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:54:16 2021

@author: andy1
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt

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
    
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()
show_images_labels(train_feature, train_label, [], 1000, 25)