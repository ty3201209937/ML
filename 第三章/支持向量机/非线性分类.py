import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def createdataset():
    df = pd.read_csv('svmdata2.csv', header=None, names=['X1', 'X2', 'y'])
    # 前五行
    print(df.head())
    # 后五行
    print(df.tail())
    return df

def svmsupport(data, c):
    # 将数据分成特征和标签
    x = data[['X1', 'X2']]
    y = data['y']
    # 创建RBF核SVM模型
    model = SVC(C=c, kernel='rbf', gamma='scale')
    # 训练模型
    model.fit(x, y)
    # 计算精度
    accuracy = model.score(x, y)
    print(f"accuracy={accuracy * 100:.2f}%")
    return model, x, y

def plot_decision_boundary(model, x, y):
    # 绘制散点图
    plt.scatter(x[y == 1]['X1'], x[y == 1]['X2'], color='blue', marker='x', label='positive')
    plt.scatter(x[y == 0]['X1'], x[y == 0]['X2'], color='orange', marker='o', label='negative')

    # 绘制决策边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格以评估模型
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.decision_function(xy).reshape(xx.shape)

    # 绘制中间的决策边界点
    ax.contour(xx, yy, Z, levels=[0], alpha=0)  # 不显示线条
    mark = True
    # 获取决策边界的坐标
    decision_boundary = np.array(np.where(np.abs(Z) < 1e-2)).T

    # 在决策边界上绘制红色点
    first_point = True
    for point in decision_boundary:
        if first_point:
            plt.plot(xx[0, point[1]], yy[point[0], 0], 'ro', markersize=1, label='boundary')
            first_point = False
        else:
            plt.plot(xx[0, point[1]], yy[point[0], 0], 'ro', markersize=1)
    # 绘制图例和标签
    plt.legend()
    plt.title(f'SVM(C={model.C}, kernel={model.kernel}) Decision Boundary')
    plt.show()

if __name__ == '__main__':
    # 使用createdataset函数读取数据
    data = createdataset()
    # 从用户输入中获取惩罚参数C的值
    c = float(input("请输入c的值: "))
    # 调用svmsupport函数训练和评估SVM模型
    model, x, y = svmsupport(data, c)
    # 调用plot_decision_boundary函数绘制决策边界
    plot_decision_boundary(model, x, y)
