import numpy as np
from models.LinearRegression import LinearRegression

import matplotlib.pyplot as plt
from utils import optimizer, RMSE, load_data

np.random.seed(2020)

# Data generation
train_data, test_data = load_data('Graduate')
x_train_data, y_train_data = train_data[0], train_data[1]
x_test_data, y_test_data = test_data[0], test_data[1]

# Hyper-parameter
_epoch=10000
_optim = 'SGD'

# ========================= EDIT HERE ========================
"""
Choose param to search. (batch_size or lr)
Specify values of the parameter to search,
and fix the other.

e.g.)
search_param = 'lr'
_batch_size = 32
_lr = [0.1, 0.01, 0.05]
"""
search_param = 'b'

if search_param == 'lr' :
    _lr = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019]
    _batch_size = 32
else :
    _batch_size = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    _lr = 0.01

#search_param이 lr인 경우에는 batch_size는 32로 고정, batch_size인 경우에는 lr을 0.01로 고정

# ============================================================


train_results = []
test_results = []
search_space = _lr if search_param == 'lr' else _batch_size
for i, space in enumerate(search_space):
    # Build model
    model = LinearRegression(num_features=x_train_data.shape[1])
    optim = optimizer(_optim)

    # Train model with gradient descent
    if search_param == 'lr':
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=_batch_size, lr=space, optim=optim)
    else:
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=space, lr=_lr, optim=optim)
    
    ################### Evaluate on train data
    # Inference
    inference = model.eval(x_train_data)

    # Assess model
    error = RMSE(inference, y_train_data)
    print('[Search %d] RMSE on Train Data : %.4f' % (i+1, error))

    train_results.append(error)

    ################### Evaluate on test data
    # Inference
    inference = model.eval(x_test_data)

    # Assess model
    error = RMSE(inference, y_test_data)
    print('[Search %d] RMSE on test data : %.4f' % (i+1, error))

    test_results.append(error)

# ========================= EDIT HERE ========================

"""
Draw scatter plot of search results.
- X-axis: search paramter
- Y-axis: RMSE (Train, Test respectively)

Put title, X-axis name, Y-axis name in your plot.

Resources
------------
Official document: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.scatter.html
"Data Visualization in Python": https://medium.com/python-pandemonium/data-visualization-in-python-scatter-plots-in-matplotlib-da90ac4c99f9
"""
if search_param == "lr" :
    plt.scatter(_lr,train_results,label = "train")
    plt.scatter(_lr,test_results, label = "test")
    
    plt.title('relationship between lr and RMSE')
    plt.xlabel('learning rate')
    plt.ylabel('RMSE')
else :
    plt.plot(_batch_size,train_results,label = "train") 
    plt.plot(_batch_size,test_results,label = "test")
    plt.title('relationship between batch size and RMSE')
    plt.xlabel('batch size')
    plt.ylabel('RMSE')

plt.legend(loc="best")
plt.show()




# ============================================================