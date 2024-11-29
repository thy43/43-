import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
  
# 假设你有一个名为df的DataFrame，其中包含你的特征数据  
# 为了示例，我们创建一个简单的DataFrame  
data = {  
    'Feature1': [1, 2, 3, 4, 5],  
    'Feature2': [5, 4, 3, 2, 1],  
    'Feature3': [2, 3, 5, 7, 11],  
    'Feature4': [7, 6, 5, 4, 3],  
    'Feature5': [11, 10, 9, 8, 7]  
}  
df = pd.DataFrame(data)  
  
# 计算特征之间的相关性矩阵  
corr_matrix = df.corr()  
  
# 使用seaborn创建热图  
plt.figure(figsize=(10, 8))  
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')  
  
# 添加标题和坐标轴标签  
plt.title('Feature Correlation Heatmap')  
plt.xlabel('Features')  
plt.ylabel('Features')  
  
# 显示热图  
plt.show()