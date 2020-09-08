import numpy as np
from models.LogisticRegression import LogisticRegression

import matplotlib.pyplot as plt
from utils import load_data, optimizer, Accuracy

np.random.seed(2020)

# Data generation
train_data, test_data = load_data('Titanic')
x_train, y_train = train_data[0], train_data[1]
x_test, y_test = test_data[0], test_data[1]

for x in x_train :
    if(x[5] > 100) : x[5] = 100
    elif(x[5] < 10) : x[5] = 1

print (x_train)