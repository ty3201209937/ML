import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import bisect

def func(x):
    return x**2 + 3*x - 10

x = np.linspace(-10, 10, 1000)
y = func(x)

gen1 = bisect(func, -10, 0)
gen2 = bisect(func, 0, 10)

print("函数的根为：", gen1, gen2)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='k')
plt.scatter([gen1, gen2], [func(gen1), func(gen2)], c='r', s=150)
plt.show()
