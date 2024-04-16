import numpy as np
import scipy as sp
from scipy.optimize import minimize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def f(x):
    return x**2+x+2
a = minimize(f,0,method='BFGS')
x = np.linspace(-10,10,1000)
y = f(x)
plt.plot(x,y)
plt.scatter(a.x,f(a.x),c='r',s=150)
plt.show()
print("最小值",a.x)