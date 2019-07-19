**update4 2019-7-14**

model.fit( )里 用validata_data 取代 validate_split
224× 224，app.yaml 及processor.py文件修改 224× 224

**update5 2019-7-15**

validate merge：将train data 和 test data都丢进去训练
batchnormalize
earlystopping by loss ，val_loss

**update6 2019-7-16**

ImageDataGenerator
save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
验证了使用效果：设训练集是300个，batch里提取的前300个是原生图片，第2次生成的300个是原生300个的变化，第3次重新生成原生300个的变化。

ReduceLROnPlateau
实验记录
1.  ResNet50
relu
args.BATCH = 32
steps_per_epoch=15,
Epoch 32/100
 - 5s - loss: 6.6469 - acc: 0.5647 - val_loss: 12.5721 - val_acc: 0.2200

 
 2.ResNet50
 relu
args.BATCH = 32
steps_per_epoch=15,
Epoch 89/100
 - 5s - loss: 0.2045 - acc: 0.9510 - val_loss: 13.2168 - val_acc: 0.1800
 
3.ResNet50
 relu
args.BATCH = 16
steps_per_epoch=60,
Epoch 63/100
 - 12s - loss: 0.0900 - acc: 0.9792 - val_loss: 13.2168 - val_acc: 0.1800
 
 4.ResNet50
 relu
args.BATCH = 16
steps_per_epoch=60,
删除了 rescale=1./255, 之前的归一化重复了？
Epoch 97/100
 - 11s - loss: 0.0698 - acc: 0.9896 - val_loss: 0.0140 - val_acc: 0.9900

 5.ResNet50
 relu
args.BATCH = 32
steps_per_epoch=150,
Epoch = 300
save_best_only = True
flyai : 86%
