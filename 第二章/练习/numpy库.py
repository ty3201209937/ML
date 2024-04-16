import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import bisect
# 1
a = np.zeros((3, 3), dtype=int)
b = np.ones((3, 3),  dtype=int)
c = np.arange(9).reshape((3, 3))
# 2
d = ['welcome to']
e = [' chongqing']
# 3
f = np.array([[1, 2, 3], [4, 5, 6]])
f1 = np.expand_dims(f, axis=0)
f2 = np.expand_dims(f, axis=1)
f3 = np.expand_dims(f, axis=2)

print('1*******************')
print("a:", a)
print("b:", b)
print("c:", c)
print('2*******************')
print(np.char.add(d, e))
print('3*******************')
print('f: ', f)
print('f1: ', f1)
print('f2: ', f2)
print('f3: ', f3)
print("线性方程的根为：",)
