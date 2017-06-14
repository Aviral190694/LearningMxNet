#Tutorial to learn Symbol Api
import mxnet as mx
import numpy as np


a = mx.symbol.Variable('A')
b = mx.symbol.Variable('B')
c = mx.symbol.Variable('C')
d = mx.symbol.Variable('D')
e = (a*b)+(c*d)
print(a,b,c,d)
print(e,type(e))
print(e.list_arguments())
print(e.list_outputs())
print(e.get_internals().list_outputs())


# Binding an array of single element
a_data = mx.nd.array([1],dtype = np.int32)
b_data = mx.nd.array([2],dtype = np.int32)
c_data = mx.nd.array([3],dtype = np.int32)
d_data = mx.nd.array([4],dtype = np.int32)

executor=e.bind(mx.cpu(), {'A':a_data, 'B':b_data, 'C':c_data, 'D':d_data})
print(executor)
e_data = executor.forward()
print(e_data, e_data[0], e_data[0].asnumpy())

#Binding an array with 1000*1000 elements

a_data = mx.nd.uniform(low=0, high=1, shape=(1000,1000))
b_data = mx.nd.uniform(low=0, high=1, shape=(1000,1000))
c_data = mx.nd.uniform(low=0, high=1, shape=(1000,1000))
d_data = mx.nd.uniform(low=0, high=1, shape=(1000,1000))

executor=e.bind(mx.cpu(), {'A':a_data, 'B':b_data, 'C':c_data, 'D':d_data})
print(executor)
e_data = executor.forward()
print(e_data, e_data[0], e_data[0].asnumpy())
