# [Apache Flink极客挑战赛——垃圾图片分类](https://tianchi.aliyun.com/competition/entrance/231743/rankingList)

队伍：SpacePro

名次：第1名

Score： 81.48

## 程序目录结构介绍

-  [garbage_image](https://github.com/LCFractal/Tianchi_garbage/tree/master/garbage_image) : java测试代码
-  [package](https://github.com/LCFractal/Tianchi_garbage/tree/master/package) : 打包用目录，包含Python代码

## 构建说明
### 1. java构建
使用maven打包程序，得到对应jar包放入package目录 
### 2. 下载预训练模型
模型使用EfficientNet作为基本模型，分别需要EfficientNetB2,B3,B4的预训练模型(下载后的h5文件放入`package/python_package`)

> **EfficientNet-B2**(https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5)

> **EfficientNet-B3**(https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5)

> **EfficientNet-B4**(https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5)


### 3. 确保package目录结构

> + package
>   + python_package
>       + model.py
>       + model_eff.py
>       + class_index.txt
>       + efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5
>       + efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5
>       + efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5
>   + requirements.txt
>   + garbage_image-1.0-SNAPSHOT.jar

### 4. 确保环境变量
> os.environ['**IMAGE_TRAIN_INPUT_PATH**']

> os.environ[**'MODEL_INFERENCE_PATH**']

## 模型说明
![Model][1]
### 1.概览
我们的模型是个融合模型，具体结构如图，分为两个部分：**主干网络**+**分类器**。

在Training阶段，经过增强的图像作为输入通过**主干网络**得到`融合feature`。使用`融合feature`及对应的标签，构建新的数据集针对**分类器**进行100个Epoch的训练，从中得到`val_acc`最大的模型，融合**主干网络**和**分类器**导出模型，作为总模型用于在Flink中进行预测。

**主干网络**部分由`EfficientNet-B2`、`EfficientNet-B3`、`EfficientNet-B4`的输出进行并联得到`联合feature`，进一步使用SVD得到`融合feature`。

**分类器**部分输入`融合feature`，然后经过`Dropout层`，经过`类SE模块`得到`加权输出`，最后通过`Dense100`作为最终的结果。

### 2.Training
任务要求在线上只有3个小时的训练时间，如果直接进行训练，本模型线上**最多训练5个Epoch**。在transfer learning中，Backbone部分不需要进行训练，因此在每次训练中，Backbone的输出是固定的，**我们不需要每个Epoch都计算Backbong的输出**。

通过上述观察，我们将训练阶段分为两个部分：**1.**主干网络抽取融合feature；**2.**使用融合feature训练分类器。

#### 1.主干网络抽取融合feature

#### 2.使用融合feature训练分类器

### 3.Prediction
预测在Flink端进行，

## 总结



  [1]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/Tianchi_garbage.png
  
