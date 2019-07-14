# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: wangyi
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
update2
'''


'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)
print('dataset.get_train_length() :',dataset.get_train_length())
print('dataset.get_all_validation_data():',dataset.get_validation_length())
'''
实现自己的网络机构
'''
num_classes = 6
sqeue = ResNet50( weights=None, input_shape=(224, 224, 3), classes= num_classes, include_top=True)

# 输出模型的整体信息
sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

'''
dataset.get_step() 获取数据的总迭代次数

'''
x_train, y_train , x_val, y_val =dataset.get_all_data()
x_train=dataset.processor_x(x_train)
y_train=dataset.processor_y(y_train)
x_val=dataset.processor_x(x_val)
y_val=dataset.processor_y(y_val)
# checkpoint = ModelCheckpoint( monitor='val_acc', mode='auto', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
best_score = 0
# for step in range(dataset.get_step()):
history = sqeue.fit(x=x_train, y=y_train,
                    # validation_data=(x_val , y_val),
                    validation_split=0.2,
                    shuffle=True,
                    batch_size=args.BATCH,
                    # callbacks = [early_stopping],
                    epochs =args.EPOCHS,
                    verbose=2)

#    print("dataset step : "+ str(step + 1) + "/" + str(dataset.get_step()))
model.save_model(sqeue, MODEL_PATH, overwrite=True)