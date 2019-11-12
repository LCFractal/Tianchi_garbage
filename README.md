# [Apache Flink极客挑战赛——垃圾图片分类](https://tianchi.aliyun.com/competition/entrance/231743/rankingList)

队伍：SpacePro

名次：第1名

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
### 概览


  [1]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/Tianchi_garbage.png
  
