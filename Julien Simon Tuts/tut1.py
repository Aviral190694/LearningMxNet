import mxnet as mx
import numpy as np

# part1
a = mx.nd.array([[1,2,3],[4,5,6]])
print(a.asnumpy())
print ("Size is",a.size)
print("Shape is",a.shape)
print("Data type is",a.dtype)

print()
# Part 2 : Changing Data type
b= mx.nd.array(([[1,2,3],[4,5,6]]), dtype = np.int32)
print(b.asnumpy())
print("Data type is :",b.dtype)


# Element Wise multiplication
print("Multiplication")
c = a*a
print("Array is",c.asnumpy())
print("Data Type is:",c.dtype)


# Dot Product
print("Dot Product")
d = a.T
print("Shape is:",d.shape)
e = mx.nd.dot(a,d)
print("Array is:",e.asnumpy())

# Creating 1000*1000 arrays 
c = mx.nd.uniform(low=0, high=1, shape=(1000,1000), ctx="cpu(0)")
d = mx.nd.normal(loc=1, scale=2, shape=(1000,1000), ctx="cpu(0)")
e = mx.nd.dot(c,d)
print(e.asnumpy())

