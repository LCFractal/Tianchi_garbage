# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
import io
import time
import shutil
import six
import tqdm
import model_eff as efn


def clearDir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


# 初始化环境变量
try:
    train_dir = os.environ['IMAGE_TRAIN_INPUT_PATH']
except:
    train_dir = ""
if len(train_dir) == 0:
    train_dir = "../../LeseNet-master/crawlers/images/"
if train_dir[-1] != "/" and train_dir[-1] != "\\":
    train_dir = train_dir + "/"
try:
    output_dir = os.environ['MODEL_INFERENCE_PATH']
except:
    output_dir = ""
if len(output_dir) == 0:
    output_dir = "../../backup/"
if output_dir[-1] != "/" and output_dir[-1] != "\\":
    output_dir = output_dir + "/"

# load data
classtxt = "/class_index.txt"
# dir_path no '/'
dir_path = os.path.dirname(os.path.abspath(__file__))
# print dir_path + classtxt
fi = io.open(dir_path + classtxt, "r", encoding="UTF8")
table = {}
CLASS_NAMES = [""] * 100
for ii in fi:
    ii = ii.replace('\n', '')
    iis = ii.split(" ")
    table[iis[0]] = int(iis[1])
    CLASS_NAMES[int(iis[1])] = iis[0]

B0 = 224
B1 = 240
B2 = 260
B3 = 300
B4 = 380
B5 = 465
BATCH_SIZE = 64
IMG_HEIGHT = 380
IMG_WIDTH = 380

if output_dir == "../../backup/":
    BATCH_SIZE = 32


# load images
def preProcessing(img):
    img[:, :, :] = img[:, :, :] / 127.5 - 1
    return img


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preProcessing
                                                                  , rotation_range=20
                                                                  , width_shift_range=0.05
                                                                  , height_shift_range=0.05
                                                                  , zoom_range=0.05
                                                                  , horizontal_flip=True
                                                                  , validation_split=0.1
                                                                  )

train_data_gen = image_generator.flow_from_directory(directory=train_dir,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes=CLASS_NAMES, subset="training")
test_data_gen = image_generator.flow_from_directory(directory=train_dir,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    classes=CLASS_NAMES, subset="validation")
all_data_gen = image_generator.flow_from_directory(directory=train_dir,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                   classes=CLASS_NAMES)

# 构建模型后端，由 EfficientNetB2,B3,B4融合
inputLayer = tf.keras.layers.Input(batch_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), name="input_1")
tf.keras.backend.set_learning_phase(0)

base_model0 = efn.EfficientNetB2(include_top=False, weights=None,
                                 input_tensor=inputLayer, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')
base_model0.trainable = False
base_model0.load_weights(dir_path + "/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5")

base_model1 = efn.EfficientNetB3(include_top=False, weights=None,
                                 input_tensor=inputLayer, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')
base_model1.trainable = False
base_model1.load_weights(dir_path + "/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5")
base_model2 = efn.EfficientNetB4(include_top=False, weights=None,
                                 input_tensor=inputLayer, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')
base_model2.trainable = False
base_model2.load_weights(dir_path + "/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5")
mergelayer = tf.keras.layers.Concatenate()([base_model0.output, base_model1.output, base_model2.output])
base_model2 = tf.keras.Model(inputs=inputLayer, outputs=mergelayer)

tf.keras.backend.set_learning_phase(1)
outdim = base_model2.output.shape.as_list()[1]

imdim = 100
mul = 6         # 数据增强参数，将输出feature扩充为数据集mul倍
if output_dir == "../../backup/":
    mul = 2
l = len(train_data_gen) * mul
train_x = np.zeros((l * BATCH_SIZE, outdim))
train_y = np.zeros((l * BATCH_SIZE, 100))
total = 0
for i in tqdm.tqdm(range(l), ncols=70):
    batch_x, batch_y = next(train_data_gen)
    train_x[total:total + len(batch_x)] = base_model2.predict(batch_x)
    train_y[total:total + len(batch_x)] = batch_y
    total = total + len(batch_x)
train_x = train_x[0:total, :]
train_y = train_y[0:total, :]
l = len(test_data_gen)
test_x = np.zeros((l * BATCH_SIZE, outdim))
test_y = np.zeros((l * BATCH_SIZE, 100))
total = 0
for i in tqdm.tqdm(range(l), ncols=70):
    batch_x, batch_y = next(test_data_gen)
    test_x[total:total + len(batch_x)] = base_model2.predict(batch_x)
    test_y[total:total + len(batch_x)] = batch_y
    total = total + len(batch_x)
test_x = test_x[0:total, :]
test_y = test_y[0:total, :]
# 计算SVD
six.print_("calc svd at %f", time.clock())
s, v, dh = np.linalg.svd(train_x, full_matrices=False)
sumall = np.sum(np.square(v))
while np.sum(np.square(v[0:imdim])) / sumall < 0.90:
    imdim = imdim + 100
six.print_("outdim=%d,imdim=%d, rate:%f" % (outdim, imdim, np.sum(np.square(v[0:imdim])) / sumall))
d = np.transpose(dh)[:, 0:imdim]
six.print_("finish svd at %f", time.clock())

# 构建新训练集（新数据集为原始图片通过backbone得到的feature）
new_train_x = np.dot(train_x, d)
new_test_x = np.dot(test_x, d)

# 正则化SVD结果
scale = np.linalg.norm(new_train_x) / np.sqrt(new_train_x.shape[0])
new_train_x = new_train_x / scale
new_test_x = new_test_x / scale
d = d / scale


def makeModel():
    # make all model used by zoo
    swish = efn.get_swish()
    merge = base_model2.output
    mergelayer = tf.keras.layers.Dense(imdim, use_bias=False, activation=swish)
    merge = mergelayer(merge)
    mergelayer.set_weights([d])
    mergelayer.trainable = False

    merge = tf.keras.layers.GaussianDropout(0.5)(merge)

    excitation = tf.keras.layers.Dense(units=imdim // 16, activation=swish)(merge)
    excitation = tf.keras.layers.Dense(units=imdim, activation='sigmoid')(excitation)
    scale = tf.keras.layers.Multiply()([merge, excitation])
    predictions = tf.keras.layers.Dense(100)(scale)

    out = tf.keras.layers.Activation("softmax")(predictions)
    model_zoo = tf.keras.Model(inputs=inputLayer, outputs=out)
    inputx = tf.keras.layers.Input(batch_shape=(None, imdim), name="input_dim")

    merge = tf.keras.layers.GaussianDropout(0.5)(inputx)

    excitation = tf.keras.layers.Dense(units=imdim // 16, activation=swish,
                                       kernel_initializer=efn.DENSE_KERNEL_INITIALIZER)(merge)
    excitation = tf.keras.layers.Dense(units=imdim, activation='sigmoid',
                                       kernel_initializer=efn.DENSE_KERNEL_INITIALIZER)(excitation)
    scale = tf.keras.layers.Multiply()([inputx, excitation])
    x = tf.keras.layers.Dense(100, kernel_initializer=efn.DENSE_KERNEL_INITIALIZER)(scale)

    predictions = tf.keras.layers.Activation("softmax")(x)
    model = tf.keras.Model(inputs=inputx, outputs=predictions)
    model.summary()
    return model, model_zoo


# model 用于python训练，model_out用于输出给zoo调用
model, model_out = makeModel()
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


minlossfunc = tf.keras.callbacks.LambdaCallback(on_epoch_end=onEpochEnd)

# 进行top model训练
if output_dir == "../../backup/":
    board = tf.keras.callbacks.TensorBoard(log_dir="E:\\天池\\backup\\logs")
    clearDir("E:\\天池\\backup\\logs")
    clearDir("E:\\天池\\backup\\plugins")
    history = model.fit(new_train_x, train_y, batch_size=BATCH_SIZE, callbacks=[reduce_lr, board, minlossfunc],
                        epochs=100, validation_data=(new_test_x, test_y))
else:
    history = model.fit(new_train_x, train_y, batch_size=BATCH_SIZE, callbacks=[reduce_lr, minlossfunc],
                        epochs=100, validation_data=(new_test_x, test_y))

six.print_(bestacc)

exit()

