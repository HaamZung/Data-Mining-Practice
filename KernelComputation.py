import csv
import math
import collections
import copy
import matplotlib.pyplot as plot


def get_dataset(filename, train_dataset, test_dataset):
    f = open("iris.csv", 'r')
    rows = f.read().split('\n')
    dataset = []
    for row in rows:
        split_row = row.split(",")
        dataset.append(split_row)

    for x in range(len(dataset) - 1):
        tmp = []
        for y in range(4):
            tmp.append(float(dataset[x][y]))
        tmp.append(dataset[x][4])
        if x % 4 != 0:
            train_dataset.append(tmp)
        else:
            test_dataset.append(tmp)


# 计算测试元组和训练元组的距离函数
def Distance(flower1, flower2, length):
    tmp = 0.0
    for i in range(length):
        tmp += math.pow(flower1[i] - flower2[i], 2)
    return math.sqrt(tmp)


# KNN算法函数
def KNN_Classification(test, train_dataset, K):
    distance = []
    for each in train_dataset:
        tmp = [Distance(each, test, 4), each[4]]
        distance.append(tmp)
    distance = sorted(distance, key=lambda distance: distance[0])[:K]
    # print distance
    collect = collections.Counter([X[1] for X in distance])
    return max(collect)


# 主函数，输入iris数据和输出结果
if __name__ == "__main__":
    train_dataset = []
    test_dataset = []
    test_dataset_bar = []
    get_dataset("iris.csv", train_dataset, test_dataset)
    print("train_database: %d" % (len(train_dataset)))
    print("test_database: %d" % (len(test_dataset)))

    test_dataset_bar = copy.deepcopy(test_dataset)
    correct_list = []
    for i in range(2, 20):
        flag = 0
        correct = 0
        for each in test_dataset:
            predict = KNN_Classification(each, train_dataset, i)
            if predict == test_dataset_bar[flag][4]:
                correct += 1
                # print "%s\t%s" % (predict, test_dataset_bar[flag][4])
            else:
                # print "%s\t%s\t++++" % (predict, test_dataset_bar[flag][4])
                pass
            flag += 1
        correct_list.append(correct * 1.0 / len(test_dataset_bar))
    print(correct_list)
    X = [i for i in range(2, 20)]
    plot.plot(X, correct_list, '.')
    plot.show()
