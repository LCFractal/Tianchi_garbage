# [Apache Flink极客挑战赛——垃圾图片分类](https://tianchi.aliyun.com/competition/entrance/231743/rankingList)

队伍：SpacePro

名次：第1名

Score： 81.48

## 程序目录结构介绍

-  [garbage_image](https://github.com/LCFractal/Tianchi_garbage/tree/master/garbage_image) : java测试代码
-  [package](https://github.com/LCFractal/Tianchi_garbage/tree/master/package) : 打包用目录，包含Python代码

## 构建说明

提供完整可直接运行压缩包：

链接： https://share.weiyun.com/5nWfsBw （密码：Ytc0）

### 1. java构建

入口为`RunZoo3.java`，使用maven打包程序，得到`garbage_image-1.0-SNAPSHOT.jar`，放入package目录。 

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

### 5. 打包并上传

将`package`文件夹打包上传天池。

## 模型说明

![Model][1]

### 1. 概览

我们的模型是个融合模型，具体结构如图，分为两个部分：**主干网络**+**分类器**。

在Training阶段，经过增强的图像作为输入通过主干网络得到`融合feature`。使用`融合feature`及对应的标签，构建新的数据集针对分类器进行100个Epoch的训练，从中得到`val_acc`最大的模型，融合主干网络和分类器导出模型，作为总模型用于在Flink中进行预测。

**主干网络**部分，由`EfficientNet-B2`、`EfficientNet-B3`、`EfficientNet-B4`的输出进行并联得到`联合feature`，进一步使用SVD得到`融合feature`。

**分类器**部分，输入`融合feature`，然后经过`Dropout层`，经过`类SE模块`得到`加权输出`，最后通过`Dense100`作为最终预测结果。

### 2. Training
任务要求在线上只有3个小时的训练时间，如果直接进行训练，本模型线上**最多训练5个Epoch**。在transfer learning中，Backbone部分不需要进行训练，因此在每次训练中，Backbone的输出是固定的，**我们不需要每个Epoch都计算Backbong的输出**。

通过上述观察，我们将训练阶段分为两个部分：1.主干网络抽取融合feature；2.使用融合feature训练分类器。

#### a. 主干网络抽取融合feature

 1) 数据准备：
```python
BATCH_SIZE = 64     # 每个Batch输入64个样本
IMG_HEIGHT = 380    # 输入图片为380*380
IMG_WIDTH = 380
# 将输入图片放缩至[-1,1]
def preProcessing(img):
    img[:, :, :] = img[:, :, :] / 127.5 - 1
    return img

# 进行一定程度的数据扩充，增加泛化能力。按0.1分割模型，用于训练和评估。
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preProcessing
                                                                  , rotation_range=20
                                                                  , width_shift_range=0.05
                                                                  , height_shift_range=0.05
                                                                  , zoom_range=0.05
                                                                  , horizontal_flip=True
                                                                  , validation_split=0.1
                                                                  )
```

2) 构建Backbone
```python
# 输入层
inputLayer = tf.keras.layers.Input(batch_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), name="input_1")

# 载入不包含top层的EfficientNet-B2，并设为不可训练
base_model0 = efn.EfficientNetB2(include_top=False, weights=None,
                                 input_tensor=inputLayer, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')
base_model0.trainable = False
base_model0.load_weights(dir_path + "/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5")

# 载入不包含top层的EfficientNet-B3，并设为不可训练
base_model1 = efn.EfficientNetB3(include_top=False, weights=None,
                                 input_tensor=inputLayer, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')
base_model1.trainable = False
base_model1.load_weights(dir_path + "/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5")

# 载入不包含top层的EfficientNet-B4，并设为不可训练
base_model2 = efn.EfficientNetB4(include_top=False, weights=None,
                                 input_tensor=inputLayer, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')
base_model2.trainable = False
base_model2.load_weights(dir_path + "/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5")

# 并联3个模型的输出
mergelayer = tf.keras.layers.Concatenate()([base_model0.output, base_model1.output, base_model2.output])
# 构建Backbone
base_model2 = tf.keras.Model(inputs=inputLayer, outputs=mergelayer)
```
3) 使用SVD，获取融合feature

我们将输入图片扩充为6倍（因为ImageDataGenerator使用了很多变换）
```python
mul = 6         # 数据增强参数，将输出feature扩充为数据集mul倍
train_x         # 存储输出的feature，用于训练
train_y         # 存储对应训练标签
test_x          # 存储输出的feature，用于测试
test_y          # 存储对应测试标签
```
计算SVD，得到融合feature，构建新的训练集
```python
s, v, dh = np.linalg.svd(train_x, full_matrices=False)
d = np.transpose(dh)[:, 0:imdim]

# 构建新训练集（新数据集为原始图片通过backbone得到的feature）
new_train_x = np.dot(train_x, d)
new_test_x = np.dot(test_x, d)

# 正则化SVD结果
scale = np.linalg.norm(new_train_x) / np.sqrt(new_train_x.shape[0])
new_train_x = new_train_x / scale
new_test_x = new_test_x / scale
d = d / scale
```
其中，`new_train_x`为新训练集，`train_y`为训练集标签；`new_test_x`为新测试集，`test_y`为测试集标签。

#### b) 使用融合feature训练分类器

构建分类器（具体参考结构图和源代码）
```python
# model 用于python训练，model_out用于输出给zoo调用
model, model_out = makeModel()
```

对已经构建的分类器进行训练
```python
# 训练参数
model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=['accuracy'])
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# 将当前val_acc最高保存为输出模型
bestacc = 0
def onEpochEnd(epoch, logs):
    global bestacc
    if bestacc <= logs["val_acc"]:
        six.print_("val_acc:%f" % (logs["val_acc"]))
        bestacc = logs["val_acc"]
        for li in [-2, -4, -5]:
            w = model.layers[li].get_weights()
            model_out.layers[li].set_weights(w)
        clearDir(output_dir + "/SavedModel/")
        tf.keras.experimental.export_saved_model(model_out, output_dir + "/SavedModel/")

# 回调，用于将当前val_acc最高保存为输出模型
bestaccfunc = tf.keras.callbacks.LambdaCallback(on_epoch_end=onEpochEnd)

# 使用融合feature进行100个epoch训练
history = model.fit(new_train_x, train_y, batch_size=BATCH_SIZE, callbacks=[reduce_lr, bestaccfunc],
                    epochs=100, validation_data=(new_test_x, test_y))
```
### 3. Prediction

预测在Flink端进行，调用完整模型，只进行简单的放缩至[-1,1]预处理
```java
// 参数设置
String modelPath = System.getenv("MODEL_INFERENCE_PATH")+"/SavedModel";
boolean ifReverseInputChannels = true;
int[] inputShape = {1, 380, 380, 3};
float[] meanValues = {127.5f, 127.5f, 127.5f};
float scale = 127.5f;
String input = "input_1";
```
Flink调用，进行预测
```java
flinkEnv.addSource(source).setParallelism(1)
    .flatMap(new ModelPredictionMapFunction2(modelPath,inputShape,ifReverseInputChannels,meanValues,scale,
                                             input,labelTable)).setParallelism(1)
    .addSink(new ImageClassSink()).setParallelism(1);
```




  [1]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/Tianchi_garbage.png
  
