#特征相关性热图
import seaborn as sns
import matplotlib.pyplot as plt

# 计算特征之间的相关性
correlation_matrix = df.corr()

# 绘制热图
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# 设置图形标题
plt.title('Feature Correlation Heatmap')

# 显示图形
plt.show()
