import numpy as np
x = np.array([[3.0,2.0,5.0,4.0],[6.0,4.0,10.0,4.0],[12.0,6.0,40.0,4.0]])

print(x)



for i in range(x.shape[1]) :
    x[:,i] = x[:,i] / np.sum(x[:,i])

print(x)