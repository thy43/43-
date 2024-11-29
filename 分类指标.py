import numpy as np  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report  
  
# 假设 y_test 是真实标签
y_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 
225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329]
# 加载SVM、CNN和KNN的预测结果  
svm_predictions = np.load('svm_predictions.npy')  
cnn_predictions = np.load('cnn_predictions.npy')  
knn_predictions = np.load('knn_predictions.npy')

# 检查各数组长度  
print("y_test length:", len(y_test))  
print("svm_predictions length:", len(svm_predictions))  
print("cnn_predictions length:", len(cnn_predictions))  
print("knn_predictions length:", len(knn_predictions))  
  
# 假设我们只需要前330个预测结果  
svm_predictions = svm_predictions[:len(y_test)]  
cnn_predictions = cnn_predictions[:len(y_test)]  
knn_predictions = knn_predictions[:len(y_test)]  
  
# 再次检查长度  
print("y_test length:", len(y_test))  
print("Truncated svm_predictions length:", len(svm_predictions))  
print("Truncated cnn_predictions length:", len(cnn_predictions))  
print("Truncated knn_predictions length:", len(knn_predictions))
  
# 确保加载的数据都是一维数组  
# 由于y_test是一个Python列表，我们将其转换为NumPy数组  
y_test = np.array(y_test)  
  
# 如果预测结果是多维的，则将它们降为一维  
if svm_predictions.ndim > 1:  
    svm_predictions = svm_predictions.ravel()  
if cnn_predictions.ndim > 1:  
    cnn_predictions = cnn_predictions.ravel()  
if knn_predictions.ndim > 1:  
    knn_predictions = knn_predictions.ravel()  
  
# 计算准确率  
svm_accuracy = accuracy_score(y_test, svm_predictions)  
cnn_accuracy = accuracy_score(y_test, cnn_predictions)  
knn_accuracy = accuracy_score(y_test, knn_predictions)  
  
# 计算精确率（平均精确率）  
svm_precision = precision_score(y_test, svm_predictions, average='macro')  
cnn_precision = precision_score(y_test, cnn_predictions, average='macro')  
knn_precision = precision_score(y_test, knn_predictions, average='macro')  
  
# 计算召回率（平均召回率）  
svm_recall = recall_score(y_test, svm_predictions, average='macro')  
cnn_recall = recall_score(y_test, cnn_predictions, average='macro')  
knn_recall = recall_score(y_test, knn_predictions, average='macro')  
  
# 计算F1分数（平均F1分数）  
svm_f1 = f1_score(y_test, svm_predictions, average='macro')  
cnn_f1 = f1_score(y_test, cnn_predictions, average='macro')  
knn_f1 = f1_score(y_test, knn_predictions, average='macro')  
  
# 打印结果进行比较  
print(f"SVM Accuracy: {svm_accuracy}, Precision: {svm_precision}, Recall: {svm_recall}, F1 Score: {svm_f1}")  
print(f"CNN Accuracy: {cnn_accuracy}, Precision: {cnn_precision}, Recall: {cnn_recall}, F1 Score: {cnn_f1}")  
print(f"KNN Accuracy: {knn_accuracy}, Precision: {knn_precision}, Recall: {knn_recall}, F1 Score: {knn_f1}")  
  
# 打印详细的分类报告  
svm_report = classification_report(y_test, svm_predictions)  
cnn_report = classification_report(y_test, cnn_predictions)  
knn_report = classification_report(y_test, knn_predictions)  
  
print("SVM Classification Report:\n", svm_report)  
print("CNN Classification Report:\n", cnn_report)  
print("KNN Classification Report:\n", knn_report)