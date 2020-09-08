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
        start_idx = 0
        end_idx = start_idx+batch_size
        y_transpose =np.transpose(np.array([y]))
           
        for i in range(epochs) :
            start_idx = 0
            end_idx = start_idx+batch_size
            y_transpose =np.transpose(np.array([y]))
            
            #샘플 추출            
            for k in range(int(x.shape[0]/batch_size)):
                x_sample = np.array([[0 for i in range(x.shape[1])]])
                y_sample = np.array([[0]])
        
                for j in range(start_idx,end_idx) :
                    sample = np.array([x[j]])
                    sample_y = np.array([y_transpose[j]])

                    x_sample = np.append(x_sample,sample,axis=0)
                    y_sample = np.append(y_sample,sample_y,axis=0)

                x_sample = np.delete(x_sample,[0,0],axis=0)
                y_sample = np.delete(y_sample,[0,0],axis=0)

                start_idx += batch_size
                if(start_idx >= x.shape[0]) : start_idx = 0

                end_idx = start_idx + batch_size
                if(end_idx > x.shape[0]) : end_idx = x.shape[0]

                #예측값
                y_predict = np.dot(x_sample,self.W)
                #오차
                error = y_predict - y_sample

                #x_transpose = np.transpose(np.array(x_sample))
                error_transpos = np.transpose(np.array(error))
                target = np.dot(error_transpos,x_sample)
                target_trnspos = np.transpose(np.array(target))
                w_diff = target_trnspos * (1/(batch_size)) 

                #업데이트
                self.W = optim.update(self.W,w_diff,lr)
                
        # ============================================================ 
    def analytic_solution(self, x, y) :
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
        
        pred = (np.dot(x,self.W))

        
        # ============================================================
        return pred
