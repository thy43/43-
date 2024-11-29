import numpy as np  
import os  
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input   # type: ignore
from tensorflow.keras.preprocessing import image   # type: ignore
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report  
import glob  
  
# 加载预训练的VGG16模型，不包括顶层的全连接层  
model = VGG16(weights='imagenet', include_top=False)  
  
# 数据集路径  
data_dir = 'F:/实用机器学习/大作业/cleaned_animal_data'  
categories = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']  
  
# 用于存储特征和标签的列表  
features = []  
labels = []  
  
# 遍历每个类别并加载图像  
for index, category in enumerate(categories):  
    path = os.path.join(data_dir, category, '*.jpg')  # 图像是jpg格式  
    for filename in glob.glob(path):  
        img = image.load_img(filename, target_size=(224, 224))  # VGG16的输入大小是224x224  
        img_array = image.img_to_array(img)  
        expanded_img_array = np.expand_dims(img_array, axis=0)  
        preprocessed_img = preprocess_input(expanded_img_array)  
        features.append(model.predict(preprocessed_img).flatten())  # 提取特征并展平  
        labels.append(index)  # 将类别标签存储为整数  
  
# 转换特征和标签为numpy数组  
features = np.array(features)  
labels = np.array(labels)  
  
# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)  
  
# 训练KNN模型  
knn = KNeighborsClassifier(n_neighbors=3)  # 设置K值，这里为3  
knn.fit(X_train, y_train)  
  
# 在测试集上进行预测并评估模型  
#y_pred = knn.predict(X_test)  
#print(classification_report(y_test, y_pred, target_names=categories))

# 在测试集上进行预测  
knn_predictions = knn.predict(X_test)  # 将预测结果存储在knn_predictions中  
  
# 评估模型并打印分类报告  
print(classification_report(y_test, knn_predictions, target_names=categories))

# 将预测结果转换为numpy数组方便后续处理
y_pred = knn_predictions

# 保存预测结果
np.save('knn_predictions.npy', y_pred)

