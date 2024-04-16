import numpy as np

a=8
b=5
c=np.bitwise_and(a,b)
print("c",c)

d=np.bitwise_or(a,b)
print("d",d)

e=np.invert(a)
print("e",e)

f=np.left_shift(a,1)
g=np.right_shift(a,1)
print("f",f)
print("g",g)
print(np.char.add(['python1'],['python2']))
print(np.char.multiply('python',3))

print(np.char.capitalize('python'))
print(np.char.title('what is python'))
print(np.char.lower('PYTHON'))
print(np.char.upper('what is python'))

h=np.array([[1,2,3],[4,5,6],[7,8,9]])
i=np.array([2,2,2])
print(np.add(h,i))
print(np.subtract(h,i))
print(np.multiply(h,i))
print(np.divide(h,i))