import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as opt

def f(x):
    return np.sin(x) + x

x = np.linspace(-10,10,1000)
y = f(x)
a = opt.bisect(f,-10,10)
print("线性方程 y = sin(x) + x 的根为：",a)
plt.plot(x,y)
plt.axhline(0, color='k')
plt.xlim(-10,10)
plt.scatter(a,f(a),c='r',s=150)
plt.show()