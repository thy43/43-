from sklearn import svm  
from sklearn.preprocessing import StandardScaler  
from PIL import Image  
import numpy as np  
import os  
import pandas as pd  # type: ignore
  
# 加载图像数据并提取特征  
def load_images_and_labels(directory):  
    images = []  
    labels = []  
    label_to_id = {}  
    id_counter = 0  
      
    # 数据集中的所有类别  
    categories = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']  
  
    for category in categories:  
        label_to_id[category] = id_counter  
        id_counter += 1  
  
    for subdir in os.listdir(directory):  
        if subdir not in categories:  
            continue  # 忽略非预定义类别的文件夹  
        subject_path = os.path.join(directory, subdir)  
        if not os.path.isdir(subject_path):  
            continue  
        for filename in os.listdir(subject_path):  
            file_path = os.path.join(subject_path, filename)  
            try:  
                image = Image.open(file_path).convert('L')  # 转换为灰度图  
                image = image.resize((64, 64))  # 调整图像大小  
                image_array = np.array(image).flatten()  # 将图像数据扁平化  
                images.append(image_array)  
                labels.append(label_to_id[subdir])  
            except Exception as e:  
                print(f"Error loading image {file_path}: {e}")  
                  
    return np.array(images), np.array(labels)  
  
# 加载数据  
data_directory = 'F:/实用机器学习/大作业/cleaned_animal_data'  # 数据集文件夹路径  
X, y = load_images_and_labels(data_directory)  
  
# 标准化特征  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  
  
# 训练SVM模型  
clf = svm.SVC(kernel='linear')  # 使用线性核  
clf.fit(X_scaled, y)  

# 标签ID到名称的映射  
label_id_to_name = {  
    0: 'Bear', 1: 'Bird', 2: 'Cat', 3: 'Cow', 4: 'Deer',  
    5: 'Dog', 6: 'Dolphin', 7: 'Elephant', 8: 'Giraffe',  
    9: 'Horse', 10: 'Kangaroo', 11: 'Lion', 12: 'Panda',  
    13: 'Tiger', 14: 'Zebra'  
}  

# 多个文件夹包含测试图像  
test_folders = [  
    'F:/实用机器学习/大作业/cleaned_animal_data/Bear',  
    'F:/实用机器学习/大作业/cleaned_animal_data/Bird',
    'F:/实用机器学习/大作业/cleaned_animal_data/Cat',
    'F:/实用机器学习/大作业/cleaned_animal_data/Cow',
    'F:/实用机器学习/大作业/cleaned_animal_data/Deer',
    'F:/实用机器学习/大作业/cleaned_animal_data/Dog',
    'F:/实用机器学习/大作业/cleaned_animal_data/Dolphin',
    'F:/实用机器学习/大作业/cleaned_animal_data/Elephant',
    'F:/实用机器学习/大作业/cleaned_animal_data/Giraffe',
    'F:/实用机器学习/大作业/cleaned_animal_data/Horse',
    'F:/实用机器学习/大作业/cleaned_animal_data/Kangaroo',
    'F:/实用机器学习/大作业/cleaned_animal_data/Lion',
    'F:/实用机器学习/大作业/cleaned_animal_data/Panda',
    'F:/实用机器学习/大作业/cleaned_animal_data/Tiger',
    'F:/实用机器学习/大作业/cleaned_animal_data/Zebra'  
]  

# 初始化一个空列表来存储SVM的预测结果  
svm_predictions = []  

# 初始化一个空列表来同时存储测试图像路径和SVM的预测结果  
# test_image_paths_and_predictions = [] 

# 遍历每个测试文件夹  
for test_folder in test_folders:  
    # 获取文件夹内所有图像文件  
    test_image_paths = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png'))]  
      
    # 遍历并预测每个图像  
    for test_image_path in test_image_paths:  
        try:  
            test_image = Image.open(test_image_path).convert('L')  
            test_image = test_image.resize((64, 64))  
            test_image_array = np.array(test_image).flatten()  
            test_image_scaled = scaler.transform([test_image_array])  
              
            # 使用模型进行预测  
            prediction = clf.predict(test_image_scaled)  

            # 将预测结果添加到svm_predictions列表中  
            svm_predictions.append(prediction[0])
              
            # 打印预测结果  
            print(f"Predicted label index for {test_image_path}: {prediction[0]}")  
            print(f"Predicted label name for {test_image_path}: {label_id_to_name[prediction[0]]}")  
        except Exception as e:  
            print(f"Error loading or predicting for test image {test_image_path}: {e}")
            
# 打印包含测试图像路径和预测结果的列表  
# for image_path, prediction in test_image_paths_and_predictions:  
#    print(f"Image Path: {image_path}, Predicted Label Index: {prediction}")

# 将预测结果转换为numpy数组方便后续处理
svm_predictions = np.array(svm_predictions)

# 保存预测结果到文件（例如使用NumPy的save功能）
np.save('svm_predictions.npy', svm_predictions)


