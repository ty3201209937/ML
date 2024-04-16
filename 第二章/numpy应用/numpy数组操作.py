import numpy as np
a= np.arange(10)
print("a=",a)
b = a.reshape(5,2)
print("b=",b)
c = np.arange(12).reshape(3,4)
print("c=",c)
d = np.transpose(c)
print("d=",d)
e = np.arange(3).reshape(1,3)
print("e=",e)
f = np.broadcast_to(e,(3,3))
print("f=",f)
g = np.array(([1,2,3],[4,5,6]))
h = np.expand_dims(g,axis = 0)
i = np.expand_dims(g,axis = 1)
print("h=",h)
print("i=",i)
j = np.array([[1,2],[3,4]])
k = np.array([[1,2],[3,4]])
l = np.concatenate((j,k), axis=0)
print("l=",l)
m = np.append(g, [[7,8,9]],axis = 0)
print("m=",m)