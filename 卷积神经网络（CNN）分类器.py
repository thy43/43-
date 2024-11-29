import os  
import numpy as np  
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore  
from tensorflow.keras.models import Sequential   # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense   # type: ignore
from tensorflow.keras.optimizers import Adam   # type: ignore
  
# 数据集路径  
data_dir = 'F:/实用机器学习/大作业/CNN_animal_data'  
  
# 图像大小  
img_width, img_height = 150, 150  
  
# 训练集和验证集的划分  
train_data_dir = os.path.join(data_dir, 'train')  
validation_data_dir = os.path.join(data_dir, 'validation')  
  
# 数据增强  
train_datagen = ImageDataGenerator(  
    rescale=1. / 255,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True)  
  
test_datagen = ImageDataGenerator(rescale=1. / 255)  
  
# 生成训练和验证数据  
train_generator = train_datagen.flow_from_directory(  
    train_data_dir,  
    target_size=(img_width, img_height),  
    batch_size=16,  
    class_mode='categorical')  
  
validation_generator = test_datagen.flow_from_directory(  
    validation_data_dir,  
    target_size=(img_width, img_height),  
    batch_size=16,  
    class_mode='categorical')  
  
# 构建CNN模型  
model = Sequential()  
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2)))  
  
model.add(Conv2D(64, (3, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2)))  
  
model.add(Flatten())  
model.add(Dense(64, activation='relu'))  
model.add(Dense(15, activation='softmax'))  # 假设有15个类别  
  
# 编译模型  
model.compile(loss='categorical_crossentropy',  
              optimizer=Adam(learning_rate=0.0001),    
              metrics=['accuracy']) 
  
# 训练模型  
epochs = 50  
batch_size = 16  
history = model.fit(  
    train_generator,  
    steps_per_epoch=train_generator.samples // batch_size,  
    epochs=epochs,  
    validation_data=validation_generator,  
    validation_steps=validation_generator.samples // batch_size)  

# 在训练模型之后，预测验证集并保存结果  
validation_steps = validation_generator.n // validation_generator.batch_size  
cnn_predictions = []  
validation_generator.reset()  # 重置验证集生成器，确保从头开始迭代 
  
for step in range(validation_steps):  
    # 获取一个批次的验证数据  
    x, _ = next(iter(validation_generator))  
    # 使用模型进行预测  
    predictions = model.predict(x)  
    # 将这个批次的预测结果添加到cnn_predictions列表中  
    cnn_predictions.extend(predictions)  
  
# 将预测结果转换为numpy数组方便后续处理  
cnn_predictions = np.array(cnn_predictions)  
  
# 保存预测结果到文件（例如使用NumPy的save功能）  
np.save('cnn_predictions.npy', cnn_predictions)

# 保存模型  
model.save('animal_classifier_model.h5')