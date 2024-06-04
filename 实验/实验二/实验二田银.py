import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
def read_dada():
    data = np.genfromtxt('太阳镜销售数据集.csv', delimiter=',')
    return data

def fit(data):
    y = data[:, 0] #销售量
    x = data[:, 1] #广告费用
    xy_sum = 0
    xx_sum = 0
    x_sum = 0
    y_sum = 0
    n = len(x)
    for i in range(n):
        x_sum += x[i]
        y_sum += y[i]
        xx_sum += x[i]*x[i]
        xy_sum += x[i]*y[i]
    x_mean = x_sum / n
    y_mean = y_sum / n
    k = (xy_sum - y_mean*x_sum)/(xx_sum - x_mean*x_sum)
    b = y_mean - k*x_mean
    return k,b
def cost(k,b,data):
    x = data[:, 1]
    y = data[:, 0]
    cost = 0
    n = len(data)
    for i in range(n):
        cost += (y[i] - k*x[i] - b) ** 2
    return cost/n
def draw(data,k,b):
    x = data[:, 1]
    y = data[:, 0]
    plt.scatter(x,y)
    plt.plot(x, x*k+b, c='r')
    plt.show()

if __name__ == '__main__' :
    data = read_dada()
    print(data)
    k, b = fit(data)
    print('k is', k)
    print('b is', b)
    cost = cost(k, b, data)
    print('cost is', cost)
    draw(data,k,b)
    x_input = int(input("输入广告费用："))
    print("销售额为：", x_input*k+b)
