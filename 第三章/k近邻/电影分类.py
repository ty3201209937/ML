import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd

def creatDataSet():
    datasets = [
        ['64', '104', '喜剧片'],
        ['57', '100', '喜剧片'],
        ['34', '81', '喜剧片'],
        ['18', '52', '喜剧片'],
        ['40', '63', '喜剧片'],
        ['55', '24', '动作片'],
        ['87', '34', '动作片'],
        ['101', '33', '动作片'],
        ['99', '21', '动作片'],
        ['54', '8', '动作片']
    ]
    labels = ['打斗镜头', '笑点镜头', '影片类型']
    return datasets, labels

def show_data():
    datasets, labels = creatDataSet()
    train_data = pd.DataFrame(datasets, columns=labels)
    print(train_data)

def draw(datasets, labels):
    # 分离数据和标签
    data = pd.DataFrame(datasets, columns=labels)
    x = data['打斗镜头'].astype(int)
    y = data['笑点镜头'].astype(int)
    types = data['影片类型']
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制数据点
    for i in range(len(data)):
        if types[i] == '喜剧片':
            plt.scatter(x[i], y[i], color='green', marker='o')
        else:
            plt.scatter(x[i], y[i], color='red', marker='s')
    # 设置标题和标签
    plt.title('电影分类散点图')
    plt.xlabel('打斗镜头')
    plt.ylabel('笑点镜头')


def clcEuclideanDis(x, y, datasets1):
    input_point = np.array([float(x), float(y)], dtype=float)
    distances = []
    for data in datasets1:
        data_point = np.array(data[:2], dtype=float)  # 提取坐标部分
        distance = np.sqrt(np.sum((input_point - data_point) ** 2))
        distances.append(distance)
    plt.scatter(x, y, color='black', marker='*')
    plt.show()


    return distances


def classify(k, distances, datasets):
    # 获取距离排序后的下标
    sorted_indices = np.argsort(distances)
    # 获取前k个下标
    nearest_indices = sorted_indices[:k]
    # 统计喜剧片和动作片的个数
    comedy_count = 0
    action_count = 0
    for index in nearest_indices:
        if datasets[index][2] == '喜剧片':
            comedy_count += 1
        else:
            action_count += 1
    return comedy_count, action_count

if __name__ == '__main__':
    datasets, labels = creatDataSet()
    show_data()
    x = int(input("请输入打斗镜头"))
    y = int(input("请输入笑点镜头"))
    # plt.scatter(x, y, color='black', marker='*')
    draw(datasets, labels)
    distances = clcEuclideanDis(x, y, datasets)
    print("distances=", distances)
    k = int(input("请输入k的值"))
    xj, dz = classify(k, distances, datasets)
    print("喜剧片：", xj)
    print("动作片：", dz)
    if xj > dz:
        print("电影类型为喜剧片")
    else:
        print("电影类型为动作片")
