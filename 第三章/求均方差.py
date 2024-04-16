import csv

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

filename = '货运量与工业总产值数据集.csv'
x = []
y = []
with open(filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))
x_sum = sum(x)
y_sum = sum(y)
x_mean = x_sum/len(x)
y_mean = y_sum/len(y)
plt.scatter(x, y)
plt.xlabel("货运量")
plt.ylabel("生产总值")
plt.title("货运量和生产总值的关系")
# legend = plt.legend(loc='upper right')
plt.show()
