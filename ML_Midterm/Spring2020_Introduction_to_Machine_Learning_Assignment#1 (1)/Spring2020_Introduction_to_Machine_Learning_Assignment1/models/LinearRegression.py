import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def numerical_solution(self, x, y, epochs, batch_size, lr, optim):

        """
        The numerical solution of Linear Regression
        Train the model for 'epochs' times with minibatch size of 'batch_size' using gradient descent.
        (TIP : if the dataset size is 10, and the minibatch size is set to 3, corresponding minibatch size should be 3, 3, 3, 1)

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )
            epochs : epochs.
            batch_size : size of the batch.
            lr : learning rate.
            optim : optimizer. (fixed to 'stochastic gradient descent' for this assignment.)

        [Output]
            None

        """

        # ========================= EDIT HERE ========================
        
        for i in range(epochs) :
            
            y_predict = np.dot(x,self.W)
            error = y - y_predict
            
            #일단 그냥 batch 방식
            w_diff = -(1/len(x)) * sum((np.dot(error,x)))
            #흠,,,벡터연산이 이렇게되나,,?
            self.W = self.W - lr*w_diff 
        

    
        # ============================================================


    def analytic_solution(self, x, y):
        """
        The analytic solution of Linear Regression
        Train the model using the analytic solution.

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )

        [Output]
            None

        """

        # ========================= EDIT HERE ========================
               
        #미분해서 푸는거 말하는 것 같은데
        #이부분 강의 듣고 다시 해봐야 할듯
        # w = A^-1 * b = (X^T * X)^-1 * (X^T * y)
        A = np.dot(np.transpose(x),x)
        b = np.dot(np.transpose(x),y)
        
        self.W = np.dot(np.linalg.inv(A),b)
        
 
        # ============================================================


    def eval(self, x):
        pred = None

        """
        Evaluation Function
        [Input]
            x : input for linear regression. Numpy array of (N, D)

        [Outputs]
            pred : prediction for 'x'. Numpy array of (N, )

        """

        # ========================= EDIT HERE ========================
        
        
        pred = np.zeros(x.shape[0])

        
        # ============================================================
        return pred
