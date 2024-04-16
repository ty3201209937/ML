import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import integrate

x = np.linspace(-1,1,10000)
f=lambda x:(1-x**2)**0.5
plt.plot(x,f(x))
plt.figure(figsize=(4,2))
plt.show()
integrate.quad(f,-1,1)
sq, err = integrate.quad(f,-1,1)
pi = sq*2
print(pi)