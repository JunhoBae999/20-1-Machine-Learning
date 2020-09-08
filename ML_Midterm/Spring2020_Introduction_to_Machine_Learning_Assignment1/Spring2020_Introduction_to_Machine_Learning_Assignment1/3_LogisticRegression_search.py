import numpy as np
from models.LogisticRegression import LogisticRegression

import matplotlib.pyplot as plt
from utils import load_data, optimizer, Accuracy

np.random.seed(2020)

# Data generation
train_data, test_data = load_data('Titanic')
x_train, y_train = train_data[0], train_data[1]
x_test, y_test = test_data[0], test_data[1]

# ========================= EDIT HERE ========================
'''
Data feature engineering.
Extract features from raw data, if you want, as you wish.

Description of each column in 'x_train' & 'x_test' is specified as follows:
    - Column 0 Pclass: Ticket class, Categorical, (1st: 0, 2nd: 1, 3rd: 2)
    - Column 1 Sex: Sex, Categorical, {male: 0, female: 1}
    - Column 2 Age: Age. Numeric, float.
    - Column 3 Siblings/Spouses Aboard: # of siblings/spouses aboard with, Numeric, integer.
    - Column 4 Parents/Children Aboard: # of parents/children aboard with, Numeric, integer.
    - Column 5 Fare: Fare of a passenger, Numeric, float.
    - Column 6 Bias: Bias initialized with 1.
'''
def feature_func_(x):
    # AS YOU WISH
    # DEFAULT: DO NOTHING.

    #Fare값 bound하기
    for data in x :
        if(data[5] >= 150) : x[5] = 6
        elif(data[5] >= 90) : x[5] = 5
        elif(data[5] >= 70) : x[5] = 4
        elif(data[5] >= 50) : x[5] = 3
        elif(data[5] >= 20) : x[5] = 2
        elif(data[5] >= 10) : x[5] = 1
        else : x[5] = 1    
    return x



# ============================================================
x_new_data = feature_func_(x_train)
assert len(x_train) == len(x_new_data), '# of data must be same.'

# Hyper-parameter
_optim = 'SGD'
_batch_size=50
# ========================= EDIT HERE ========================
'''
Tuning hyper-parameters.
Here, tune two kinds of hyper-parameters, 
# of epochs (_epoch) and learning_rate (_lr).

'''
_epoch=  10000
_lr = 0.002

# ============================================================

# Build model
model = LogisticRegression(num_features=x_new_data.shape[1])
optimizer = optimizer(_optim)

# Solve
print('Train start.')
model.fit(x=x_new_data, y=y_train, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optimizer)
print('Trained done.')

# Inference
print('Predict on test data')
inference = model.eval(feature_func_(x_test))

# Assess model
error = Accuracy(inference, y_test)
print('Accuracy on test data : %.4f' % error)
