# ML_models
> machine learning models

## mnist demo
> 手寫數字辨識

* mnist_demo_model：784 -> 512 -> 10
* mnist_demo_model_dropout：784 -> 1024 (Dropout:0.2) -> 10

## house price
> 房價預測模型

使用Jupyter執行，並用TensorBoard顯示即時的loss變化

* best-model-1：對照組
* best-model-2：減少模型大小
* best-model-3：加入L2 Regularization
* best-model-4：加入Dropout，Dropout rate=0.2

## CUDA & cuDNN Requirement
* CUDA Toolkit 11.0 (May 2020)  
* cuDNN v8.0.4 (September 28th, 2020), for CUDA 11.0

rename cusolver64_10.dll -> cusolver64_11.dll