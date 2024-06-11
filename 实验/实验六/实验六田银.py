import matplotlib
from sklearn.svm import LinearSVC, SVC

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt    #导入matplotlib库
import matplotlib as mpl
import operator                    #导入operator库
import numpy as np

def createDataSet():               #创建数据集
    data = np.genfromtxt('iris.csv', delimiter=',', skip_header=1)#指定第一行为标题行，跳过
    x = data[:,1:3]
    y = data[:,-1]
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 2
    return x,y

def draw(group,labels):           #定义draw函数，画样本散点
    mpl.rcParams['font.family'] = ['sans-serif']
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False     #设置显示字体
    for i in range(len(group)):   #for循环遍历所有样本点
        if labels[i] == 0:
           plt.scatter(group[i][0], group[i][1], marker='o', color='blue', s=10)
        else:
            plt.scatter(group[i][0], group[i][1], marker='o', color='orange', s=10)
    plt.show()

def svmsupport(x, y):
    svc = SVC()
    svc.fit(x, y)
    return svc

def plot_decision_boundary(model, x, y):
    mpl.rcParams['font.family'] = ['sans-serif']
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False  # 设置显示字体
    # 绘制所有样本点
    for i in range(len(x)):
        if y[i] == 0:
            plt.scatter(x[i][0], x[i][1], marker='o', color='blue', s=10)
        else:
            plt.scatter(x[i][0], x[i][1], marker='o', color='orange', s=10)

    # 绘制决策边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # 创建网格以评估模型
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.decision_function(xy).reshape(xx.shape)

    # 获取决策边界的坐标
    decision_boundary = np.array(np.where(np.abs(Z) < 1e-2)).T

    # 在决策边界上绘制红色点
    first_point = True
    for point in decision_boundary:
        if first_point:
            plt.plot(xx[0, point[1]], yy[point[0], 0], 'ro', markersize=2, label='boundary')
            first_point = False
        else:
            plt.plot(xx[0, point[1]], yy[point[0], 0], 'ro', markersize=2)
    # 绘制图例和标签
    plt.legend()
    plt.title(f'SVM(C={model.C}) Decision Boundary')
    plt.show()


if __name__=="__main__":
    x,y=createDataSet()   #调用createDataSet函数，得到group和labels
    print('x=', x)                   #打印group
    print('y=', y)                  #打印labels
    # draw(x, y)
    svc = svmsupport(x, y)
    plot_decision_boundary(svc,x,y)