# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse

from keras.applications import ResNet50
from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D
from keras.models import Sequential
from model import Model
from path import MODEL_PATH
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint


'''
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=100, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

'''
实现自己的网络机构
'''
num_classes = 6
sqeue = ResNet50( weights=None, input_shape=(224, 224, 3), classes= num_classes, include_top=True)

# 输出模型的整体信息
sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''
dataset.get_step() 获取数据的总迭代次数

'''
# checkpoint = ModelCheckpoint( monitor='val_acc', mode='auto', save_best_only=True)
best_score = 0
for step in range(dataset.get_step()):
    x_train, y_train = dataset.next_train_batch()
    x_val, y_val = dataset.next_validation_batch()
    history = sqeue.fit(x_train, y_train, validation_data=(x_val , y_val),
                        batch_size=args.BATCH,
                        # callback = checkpoint,
                        epochs =args.EPOCHS,
                        verbose=1)

    print(str(step + 1) + "/" + str(dataset.get_step()))
model.save_model(sqeue, MODEL_PATH, overwrite=True)