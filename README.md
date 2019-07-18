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
