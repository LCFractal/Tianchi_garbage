# [Apache Flink极客挑战赛——垃圾图片分类](https://tianchi.aliyun.com/competition/entrance/231743/rankingList)

队伍：SpacePro

名次：第1名

Score： 81.48

## 任务分析
![Introduction][1]
任务对模型的使用环境有着诸多的限制，因此我们有以下思考：

 1. 模型可以提供经过预训练的模型，我们可以本地收集数据集提前预训练模型进一步在网上微调。
 2. 任务限制训练时间在三小时以内，不足以进行充分的训练，可以使用迁移学习在短期内获得较高的结果。
 3. 模型足够高效，较小的参数量，足够快的收敛速度，足够快的推理速度。
 4. 在训练阶段，让模型快速收敛或者让模型在同样时间训练更多轮次。
 5. Flink端限制单张图片500ms，因此需要尽可能简单的预处理与尽可能高效的模型。
 6. 因为测试阶段图片不能进行过多的处理，训练得到的模型需要有足够的适应性，我们需要数据增强方案来增加模型适应性。

## 模型分析

![Model Analysis][2]

在比赛期间我们所使用模型的路线图，这些模型在提升精度的同时依旧保证了效率。最终我们得到了精度最高的模型，达到了**81.48%**。

以下我们介绍**3EfficientNets+attention+SVD**模型的思路：
1. 模型不需要在本地进行预训练，只需要经过ImageNet预训练的EfficientNet模型。
2. 使用EfficientNet-B2,B3,B4三个模型的融合模型作为主干网络，增加模型宽容度。
3. EfficientNet是一个参数少、效率高、表现好的分类网络，与ResNet-50效率类似，表现远远超过ResNet-50。
4. **我们提出主干网络+分类器的分离模型，大大提高训练速度，保证能够训练出足够优秀的分类器。**
5. **我们提出SVD+attention方法，大大增加了3EfficientNets的精度，同时降低了分类器的参数量。**
6. Flink端的预处理仅仅将输入图片放缩至[-1,1]，保证足够快的测试速度。
7. 使用常用的数据增强方式(旋转、平移、反转、放缩)进一步增加模型的宽容度。


## 模型构建

![Model][3]

### 1. 概览

我们的模型是个融合模型，具体结构如图，分为两个部分：**主干网络**+**分类器**。

在Training阶段，经过增强的图像作为输入通过主干网络得到`融合feature`。使用`融合feature`及对应的标签，构建新的数据集针对分类器进行100个Epoch的训练，从中得到`val_acc`最大的模型，融合主干网络和分类器导出模型，作为总模型用于在Flink中进行预测。

**主干网络**部分，由`EfficientNet-B2`、`EfficientNet-B3`、`EfficientNet-B4`的输出进行并联得到`联合feature`，进一步使用SVD得到`融合feature`。

**分类器**部分，输入`融合feature`，然后经过`Dropout层`，经过`Attention模块`得到`加权输出`，最后通过`Dense100`作为最终预测结果。

### 2. Training
任务要求在线上只有3个小时的训练时间，如果直接进行训练，本模型线上**最多训练5个Epoch**。在transfer learning中，Backbone部分不需要进行训练，因此在每次训练中，Backbone的输出是固定的，**我们不需要每个Epoch都计算Backbong的输出**。

通过上述观察，我们将训练阶段分为两个部分：1.主干网络抽取融合feature；2.使用融合feature训练分类器。

#### a) 主干网络抽取融合Feature

![Backbone][4]

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

> 我们将输入图片扩充为6倍（因为ImageDataGenerator使用了很多变换）

```python
mul = 6         # 数据增强参数，将输出feature扩充为数据集mul倍
train_x         # 存储输出的feature，用于训练
train_y         # 存储对应训练标签
test_x          # 存储输出的feature，用于测试
test_y          # 存储对应测试标签
```
![SVD][5]

> 计算SVD，得到融合feature，构建新的训练集

```python
s, v, dh = np.linalg.svd(train_x, full_matrices=False)
d = np.transpose(dh)[:, 0:imdim]

# 构建新训练集（新数据集为原始图片通过backbone得到的feature，再乘以svd的特征矩阵）
new_train_x = np.dot(train_x, d)
new_test_x = np.dot(test_x, d)

# 正则化SVD结果（由于SVD直接乘出来数值较大，使得最后一层也输出较大，使softmax过于“自信”）
scale = np.linalg.norm(new_train_x) / np.sqrt(new_train_x.shape[0])
new_train_x = new_train_x / scale
new_test_x = new_test_x / scale
d = d / scale
```

> 其中，`new_train_x`为新训练集，`train_y`为训练集标签；`new_test_x`为新测试集，`test_y`为测试集标签。

#### b) 使用融合feature训练分类器

![Classifier][6]

> 构建分类器
```python
# model 用于python训练，model_out用于输出给zoo调用
model, model_out = makeModel()
```

> 对已经构建的分类器进行训练

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

![Prediction][7]

在预测阶段我们将之前的主干网络和分类器合并为一个网络，需要注意的是SVD层我们根据SVD计算结果构造一个全连接层来达到SVD层的效果。这里有一个假设，训练集和测试集为同一个分布，处于同一个特征空间下，这样我们可以认为通过训练集计算出的SVD参数已经可以代表测试集的参数。

> 预测在Flink端进行，为了保证运行效率只进行简单的放缩至[-1,1]预处理

```java
// 参数设置
String modelPath = System.getenv("MODEL_INFERENCE_PATH")+"/SavedModel";
boolean ifReverseInputChannels = true;
int[] inputShape = {1, 380, 380, 3};
float[] meanValues = {127.5f, 127.5f, 127.5f};
float scale = 127.5f;
String input = "input_1";
```
> Flink调用，进行预测
```java
flinkEnv.addSource(source).setParallelism(1)
    .flatMap(new ModelPredictionMapFunction2(modelPath,inputShape,ifReverseInputChannels,meanValues,scale,
                                             input,labelTable)).setParallelism(1)
    .addSink(new ImageClassSink()).setParallelism(1);
```

## 实验分析
![Analysis][8]

我们的模型在3EfficientNets的基础上进一步引入了SVD和attention结构，消融实验表明了我们引入的结构对于模型的提升是十分明显的，特别是在SVD和attention被同时使用时。

我们从直观上分析：

 - SVD筛选了重要特征剔除了次要特征使得分类器具有更少的参数，小的参数空间更容易找到最优的结果
 - attention同样起到了筛选特征重要性的作用，简单的拼接特征不进行筛选的话会增加搜索难度；
 - SVD+attention，在降低了特征数量之后，attention可以更容易的筛选特征，从更少的选项中更容易找到重要的特征，从而使得分类更加有效。

## 总结展望

### 总结

 - 构造3EfficientNets+attention+SVD融合模型，达到81.48%精度。
 - 使用3EfficientNets由，EfficientNet-B2,B3,B4组合而成，这是Google 2019年提出的目前最高效的模型。
 - 提出SVD+attention结构，降低参数量，显著提升精度。
 - 提出主干网络+分类器，分离训练，更加高效的训练分类器。
 - 使用旋转、平移、放缩、翻转进行数据增强，增加模型宽容度。
 - Flink仅仅进行最简单的数据预处理，保证效率。

### 展望

 - SVD+attention或许是一个新的针对于迁移学习的有效模块，同时SVD应该可以端到端的建立在网络中。
 - 使用更加有针对性的数据增强能够进一步提升模型的性能。
 - 目前的分类器对于特征的利用依然简单，或许可以构建更加有效利用特征的分类器提升性能。


  [1]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/PPT/02.PNG
  [2]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/PPT/03.PNG
  [3]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/PPT/04.PNG
  [4]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/PPT/11.PNG
  [5]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/PPT/10.PNG
  [6]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/PPT/12.PNG
  [7]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/PPT/13.PNG
  [8]: https://raw.githubusercontent.com/LCFractal/Tianchi_garbage/master/img/PPT/14.PNG
