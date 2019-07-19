# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: wangyi

update4 尝试新增一些东西
validata_data 取代 split
512 × 384 ，输入图片再改改

update5
validate merge
batchnormalize
earlystopping by loss

update6
ImageDataGenerator

update7
妈的！删除了 rescale=1./255, 之前的归一化重复了。

"""

import argparse
from keras.applications import ResNet50,VGG16
from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D,ZeroPadding2D,BatchNormalization
from keras.models import Sequential
from model import Model
from path import MODEL_PATH
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import SGD,adam
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import sys
import os

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)
'''
dataset获取train & test数据的总迭代次数

'''
x_train, y_train , x_val, y_val =dataset.get_all_processor_data()

print('dataset.get_train_length() :',dataset.get_train_length())
print('dataset.get_all_validation_data():',dataset.get_validation_length())
'''
实现自己的网络机构
'''
num_classes = 6
sqeue =ResNet50( weights=None, input_shape=(224, 224, 3), classes= num_classes, include_top=True)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


# 输出模型的整体信息
sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.check( MODEL_PATH, overwrite=True)
# 模型保存的路径
MODEL_PATH_FILE = os.path.join(sys.path[0], 'data', 'output', 'model', 'model.h5')
savebestonly = ModelCheckpoint( filepath =MODEL_PATH_FILE, monitor='val_loss', mode='auto', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=20 ,verbose=1)
xuexilv = ReduceLROnPlateau(monitor='loss',patience=20, verbose=1)
best_score = 0

x_train_and_x_val = np.concatenate((x_train, x_val),axis=0)
y_train_and_y_val= np.concatenate((y_train , y_val),axis=0)
# print('x_train_and_x_val[0]', x_train_and_x_val[0])

# 采用数据增强ImageDataGenerator
datagen= ImageDataGenerator(
    # rescale=1./255,
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0.02,
    horizontal_flip=True,
    vertical_flip=True

)
# datagen.fit(x_train_and_x_val)
# save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
data_iter = datagen.flow(x_train_and_x_val, y_train_and_y_val, batch_size=args.BATCH , save_to_dir = None)
# 验证集可以也写成imagedatagenerator

print('x_train_and_x_val.shape :', x_train_and_x_val.shape)
'''
history = sqeue.fit(x=x_train_and_x_val, y=y_train_and_y_val,
                    validation_data=(x_val , y_val),
                    # validation_split=0.2,
                    shuffle=True,
                    batch_size=args.BATCH,
                    callbacks = [early_stopping],
                    epochs =args.EPOCHS,
                    verbose=2)
'''

history = sqeue.fit_generator(
    data_iter,
    steps_per_epoch=75,
    validation_data=(x_val , y_val),
    # validation_steps=5,
    # callbacks = [early_stopping,xuexilv],
    callbacks = [ savebestonly, xuexilv],
    epochs =args.EPOCHS,
    verbose=2,
    workers=24
    # use_multiprocessing=True
)


#    print("dataset step : "+ str(step + 1) + "/" + str(dataset.get_step()))
# model.save_model(sqeue, MODEL_PATH, overwrite=True)