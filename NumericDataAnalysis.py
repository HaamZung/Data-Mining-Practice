import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

# 载入iris数据集
data = np.loadtxt("C:/Users/xianzong24/Desktop/数据挖掘/magic04.txt",
                  delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
print("Row Data（原始数据集）:\n", data)

col_num = np.size(data, axis=1)  # 获取数据总列数
row_num = np.size(data, axis=0)  # 获取数据行数

mean_vector = np.mean(data, axis=0).reshape(col_num, 1)  # 计算Mean Vector
print("Mean Vector（均值向量）:\n", mean_vector)

t_mean_vector = np.transpose(mean_vector)  # 转置Mean Vector
centered_data_matrix = data - (1 * t_mean_vector)  # 计算Centered Data Matrix
print("Centered Data Matrix（中心数据矩阵）:\n", centered_data_matrix, "\n")

t_centered_data_matrix = np.transpose(centered_data_matrix)  # 转置Centered Data Matrix
covariance_matrix_inner = (1 / row_num) * np.dot(t_centered_data_matrix, centered_data_matrix)
# 计算样本协方差矩阵
print("以中心数据矩阵列为内乘积的样本协方差矩阵：\n",
      covariance_matrix_inner, "\n")


# 计算中心数据点的和
def sum_of_centered_points():
    sum = np.zeros(shape=(col_num, col_num))
    for i in range(0, row_num):
        sum += np.dot(np.reshape(t_centered_data_matrix[:, i], (-1, 1)),
                      np.reshape(centered_data_matrix[i, :], (-1, col_num)))
    return sum


covariance_matrix_outer = (1 / row_num) * sum_of_centered_points()
print("样本协方差矩阵作为中心数据点之间的外积：\n",
      covariance_matrix_outer, "\n")

vector1 = np.array(centered_data_matrix[:, 1])
vector2 = np.array(centered_data_matrix[:, 2])


# 计算属性向量的单位向量
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


# 计算属性向量之间的夹角
def angle_between(v1, v2):
    u_v1 = unit_vector(v1)
    u_v2 = unit_vector(v2)
    return np.arccos(np.clip(np.dot(u_v1, u_v2), -1.0, 1.0))


correlation = math.cos(angle_between(vector1, vector2))  # 计算各属性间的相关性
print("属性1和2之间的相关性： %.5f" % correlation, "\n")

variance_vector = np.var(data, axis=0)  # 创建一个方差向量
max_var = np.max(variance_vector)  # 计算最大方差
min_var = np.min(variance_vector)  # 计算最小方差

for i in range(0, col_num):  # 找出最大方差向量的索引
    if variance_vector[i] == max_var:
        max_var_index = i

for i in range(0, col_num):  # 找出最小方差向量的索引
    if variance_vector[i] == min_var:
        min_var_index = i

print("Max variance = %.3f (Attribute %d )" % (max_var, max_var_index))
print("Min variance = %.3f (Attribute %d )\n" % (min_var, min_var_index))

covariance_matrix = np.cov(data, rowvar=False)  # 计算协方差矩阵
max_cov = np.max(covariance_matrix)  # 找出协方差矩阵中的最大值
min_cov = np.min(covariance_matrix)  # 找出协方差矩阵中的最小值

# 利用for循环找出最大最小值的索引
for i in range(0, col_num):
    for j in range(0, col_num):
        if covariance_matrix[i, j] == max_cov:
            max_cov_attr1 = i
            max_cov_attr2 = j

for i in range(0, col_num):
    for j in range(0, col_num):
        if covariance_matrix[i, j] == min_cov:
            min_cov_attr1 = i
            min_cov_attr2 = j

print("Max Covariance = %.3f (Between Attribute %d and %d)" % (max_cov, max_cov_attr1, max_cov_attr2))
print("Min Covariance = %.3f (Between Attribute %d and %d)" % (min_cov, min_cov_attr1, min_cov_attr2))

df = pd.DataFrame(data[:, 1])  # 创建绘图数据框
plt.show(plt.scatter(data[:, 1], data[:, 2], c=("red", "yellow")))  # 绘制属性散点图
plt.show(df.plot(kind="density"))  # 绘制概率密度函数
