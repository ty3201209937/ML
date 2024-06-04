import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def creatdataset():
    points = np.genfromtxt("人工智能技术专业学生身高体重数据集 (1).csv", delimiter=",")  #1 文件名错了
    return points

def draw(points):
    plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    x = points[:, 0]   #2 x是身高
    y = points[:, 1]   #3 y是体重
    plt.scatter(x, y)
    plt.title(r'人工智能技术专业学生身高与体重预测')
    plt.xlabel('学生身高(cm)')
    plt.ylabel('学生体重(kg)')
    plt.show()

def fit(points):
    N = len(points)                 #4 求人数
    x_bar = np.sum(points[:, 1])   #5 y值的和
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0
    x_sum = 0
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        x_sum += x
        sum_yx += x*y            #6 x*y的和
        sum_x2 += x ** 2                      #7 平方
    k = (sum_yx-(x_bar/N)*x_sum) / (sum_x2 - (x_sum/N)*x_sum)    #8 k=(x_y_sum - y_mean * x_sum) / (x_2_sum - x_mean * x_sum)
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        sum_delta += (y - k * x)          #9  b = y_mean - k*x_mean
    b = sum_delta/N                      #10  sum_delta应除以人数
    return k, b

def compute_cost(k,b,points):
    total_cost = 0
    N = len(points)
    for i in range(N):
        x = points[i, 0]                     #11x为身高
        y = points[i, 1]                     #12y为体重
        total_cost += (y-k*x-b)**2
    return total_cost/N


if __name__ == '__main__':
    points=creatdataset()
    draw(points)
    k, b = fit(points)
    cost = compute_cost(k, b, points)        #13 数据集为points
    print("k is :", k)
    print("b is :", b)
    print("cost is: ",cost)

    x=points[:,0]
    pred_y = k * np.array(x) + b                         #14 计算方式应为数组点乘，不能直接乘
    plt.plot(x, pred_y, c='r')
    draw(points)
    plt.show()

    height=input('请输入您的身高：')
    weight=k*int(height)+b                     #15 输入体重的变量应为height
    print('根据人工智能技术专业学生身高与体重线性回归模型，预测您的体重应是：')
    print(weight)