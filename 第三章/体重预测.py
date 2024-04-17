import csv
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pylab import mpl

def import_modules():
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    x = [float(row[0]) for row in data]
    y = [float(row[1]) for row in data]
    return x, y

def calculate_loss(x, y):
    x_sum = sum(x)
    y_sum = sum(y)
    x_mean = x_sum / len(x)
    y_mean = y_sum / len(y)
    x_2_sum = sum(x_i ** 2 for x_i in x)
    x_y_sum = sum(x[i] * y[i] for i in range(len(x)))
    w = (x_y_sum - y_mean * x_sum) / (x_2_sum - x_mean * x_sum)
    b = y_mean - w * x_mean
    D = sum((y[i] - w * x[i] - b) ** 2 for i in range(len(x)))
    return w, b, D

def plot_scatter_and_line(x, y, w, b):
    plt.scatter(x, y)
    plt.plot(x, w * np.array(x) + b, c='r')
    plt.xlabel("身高")
    plt.ylabel("体重")
    plt.title("身高和体重的关系")
    plt.show()

if __name__ == "__main__":
    import_modules()
    filename = '人工智能技术专业学生身高体重数据集.csv'
    x, y = read_data(filename)
    w, b, loss = calculate_loss(x, y)
    print("均方差损失函数值为：", loss)
    plot_scatter_and_line(x, y, w, b)
    your = input("请输入你的身高：")
    print(w*float(your)+b)

