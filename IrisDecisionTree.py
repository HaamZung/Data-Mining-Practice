import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
# print(iris)

test_idx = [11, 17, 117, 51]
# 随机抽取数据集中的数据作预测

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# 把抽取样本数据输入决策树函数中
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print("正确结果：\n", test_target)
print("预测结果:\n", clf.predict(test_data))
print(iris.feature_names, "->", iris.target_names)
print(test_data, "->", test_target)
