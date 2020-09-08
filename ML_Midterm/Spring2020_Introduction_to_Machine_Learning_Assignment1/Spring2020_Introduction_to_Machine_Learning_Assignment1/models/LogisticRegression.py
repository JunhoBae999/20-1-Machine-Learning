import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def fit(self, x, y, epochs, batch_size, lr, optim):

        """
        The optimization of Logistic Regression
        Train the model for 'epochs' times with minibatch size of 'batch_size' using gradient descent.
        (TIP : if the dataset size is 10, and the minibatch size is set to 3, corresponding minibatch size should be 3, 3, 3, 1)

        [Inputs]
            x : input for logistic regression. Numpy array of (N, D)
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

                #오차
                argument = np.transpose(np.array(np.dot(x_sample,self.W)))

                sigmoid_x = self._sigmoid(argument)

                transpose_sigmoid_x = np.transpose(np.array(sigmoid_x))
                
                error = transpose_sigmoid_x - y_sample

                error_transpose = np.transpose(np.array(error))

                target = np.dot(error_transpose,x_sample)
                target_trnspos = np.transpose(np.array(target))
                w_diff = target_trnspos * (1/batch_size) 
               
                #x_transpose = np.transpose(np.array(x_sample))
                #업데이트
                self.W = optim.update(self.W,w_diff,lr)
        
        # ============================================================
    
    def _sigmoid(self, x):


        """
        Apply sigmoid function to the given argument 'x'.

        [Inputs]
            x : Input of sigmoid function. Numpy array of arbitrary shape.

        [Output]
            sigmoid: Output of sigmoid function. Numpy array of same shape with 'x'.

        """
        sigmoid = None
        # ========================= EDIT HERE ========================
        sigmoid = 1/(1+np.exp(-x))
        # ============================================================
        return sigmoid

    def eval(self, x, threshold=0.5):
        pred = None

        """
        Evaluation Function
        [Input]
            x : input for logistic regression. Numpy array of (N, D)

        [Outputs]
            pred : prediction for 'x'. Numpy array of (N, )
                    Pred = 1 if probability > threshold 
                    Pred = 0 if probability <= threshold 
        """

        # ========================= EDIT HERE ========================

        # Temp
        pred = np.zeros((x.shape[0]))
        
        temp = np.dot(x,self.W)
        temp = self._sigmoid(temp)
        i=0
        for elements in temp :
            if(elements > threshold) : pred[i] += 1
            i+=1
        # ============================================================
        return pred
