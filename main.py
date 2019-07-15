# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: wangyi

update4 尝试新增一些东西
validata_data 取代 split
512 × 384 ，输入图片再改改
"""

import argparse

from keras.applications import ResNet50,VGG16
from flyai.dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D,ZeroPadding2D
from keras.models import Sequential
from model import Model
from path import MODEL_PATH
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
from keras.optimizers import SGD

'''
update2
'''


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
x_train, y_train , x_val, y_val =dataset.get_all_data()
x_train=dataset.processor_x(x_train)
y_train=dataset.processor_y(y_train)
x_val=dataset.processor_x(x_val)
y_val=dataset.processor_y(y_val)
print('dataset.get_train_length() :',dataset.get_train_length())
print('dataset.get_all_validation_data():',dataset.get_validation_length())
'''
实现自己的网络机构
'''
num_classes = 6
sqeue = Sequential()
sqeue.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(384, 512, 3)))
sqeue.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
sqeue.add(MaxPooling2D(pool_size=(2, 2)))
# sqeue.add(Dropout(0.25))

sqeue.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
sqeue.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
sqeue.add(MaxPooling2D(pool_size=(2, 2)))
# sqeue.add(Dropout(0.25))

sqeue.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
sqeue.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
sqeue.add(MaxPooling2D(pool_size=(2, 2)))
# sqeue.add(Dropout(0.25))

sqeue.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
sqeue.add(MaxPooling2D(pool_size=(2, 2)))

sqeue.add(Flatten())
sqeue.add(Dense(1024, activation='relu'))
sqeue.add(Dropout(0.5))
sqeue.add(Dense(1024, activation='relu'))
sqeue.add(Dropout(0.5))
sqeue.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


# 输出模型的整体信息
sqeue.summary()

sqeue.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# checkpoint = ModelCheckpoint( monitor='val_acc', mode='auto', save_best_only=True)
early_stopping = EarlyStopping(monitor='loss', patience=10 ,verbose=1)
best_score = 0
# for step in range(dataset.get_step()):
history = sqeue.fit(x=x_train, y=y_train,
                    validation_data=(x_val , y_val),
                    # validation_split=0.2,
                    shuffle=True,
                    batch_size=args.BATCH,
                    # callbacks = [early_stopping],
                    epochs =args.EPOCHS,
                    verbose=2)

#    print("dataset step : "+ str(step + 1) + "/" + str(dataset.get_step()))
model.save_model(sqeue, MODEL_PATH, overwrite=True)