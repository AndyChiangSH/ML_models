# ML_models
machine learning models

## Models
### mnist demo
> 手寫數字辨識

* mnist_demo_model：Input(784) -> Dense(512) -> Output(10)
* mnist_demo_model_dropout：Input(784) -> Dense(1024) (Dropout:0.2) -> Output(10)
* mnist_demo_model3(CNN)：Input(n, 28, 28, 1) -> Convolution(15, (3, 3)) -> MaxPooling(2, 2) -> Convolution(30, (3, 3)) -> MaxPooling(2, 2) -> Flatten -> Dense(100) -> Output(10)

### house price
> 房價預測模型

使用Jupyter執行，並用TensorBoard顯示即時的loss變化

* best-model-1：對照組
* best-model-2：減少模型大小
* best-model-3：加入L2 Regularization
* best-model-4：加入Dropout，Dropout rate=0.2

### pokemon combat
> 寶可夢對戰勝率預測

* best-model-1：屬性以數值表示
* best-model-2：屬性以one hot表示
* two_pokemon_combat：隨機抽樣兩隻寶可夢對戰

## CUDA & cuDNN Requirement
* CUDA Toolkit 11.0 (May 2020)  
* cuDNN v8.0.4 (September 28th, 2020), for CUDA 11.0 -> v8.1.0

rename cusolver64_10.dll -> cusolver64_11.dll

## TensorBoard

```
cd 專案路徑
tensorboard --logdir logs
```
