import mxnet as mx
import numpy as num

a = mx.nd.array([[1,2,3],[4,5,6]])
print(a.asnumpy())
print ("Size is",a.size)
print("Shape is",a.shape)
print("Data type is",a.dtype)


