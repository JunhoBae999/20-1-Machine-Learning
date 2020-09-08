import time
import numpy as np

class SoftmaxClassifier:
    def __init__(self, num_features, num_label):
        self.num_features = num_features
        self.num_label = num_label
        self.W = np.zeros((self.num_features, self.num_label))

    def train(self, x, y, epochs, batch_size, lr, optimizer):
        """
        N : # of training data
        D : # of features
        C : # of classes

        Inputs:
        x : (N, D), input data
        y : (N, )
        epochs: (int) # of training epoch to execute
        batch_size : (int) # of minibatch size
        lr : (float), learning rate
        optimizer : (Class) optimizer to use

        Returns:
        None

        Description:
        Given training data, hyperparameters and optimizer, execute training procedure.
        Weight should be updated by minibatch (not the whole data at a time)
        Procedure for one epoch is as follow:
        - For each minibatch
            - Compute probability of each class for data
            - Compute softmax loss
            - Compute gradient of weight
            - Update weight using optimizer
        * loss of one epoch = refer to the loss function in the instruction.
        """
        num_data, num_feat = x.shape
        num_batches = int(np.ceil(num_data / batch_size))

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            epoch_loss = 0.0
            # ========================= EDIT HERE ========================
            start_idx = 0
            end_idx = num_batches
            while(True) :
                if end_idx > num_data :
                    end_idx = num_data
                if start_idx > num_data :
                    break

                #mini batch data
                x_sample = x[start_idx:end_idx,:]
                y_sample = y[start_idx:end_idx]

                #comput probability for data,softmax loss
                prob,loss = self.forward(x_sample,y_sample)
                epoch_loss += loss
                grad = self.compute_grad(x_sample,y_sample,self.W,prob)

                self.W = optimizer.update(self.W,grad,lr)

                start_idx += num_batches
                end_idx += num_batches

                if end_idx == num_data :
                    break
            # ============================================================
            epoch_elapsed = time.time() - epoch_start
            print('epoch %d, loss %.4f, time %.4f sec.' % (epoch, epoch_loss, epoch_elapsed))

    def forward(self, x, y):
        """
        N : # of minibatch data
        D : # of features

        Inputs:
        x : (N, D), input data 
        y : (N, ), label for each data

        Returns:
        prob: (N, C), probability distribution over classes for N data
        softmax_loss : float, softmax loss for N input

        Description:
        Given N data and their labels, compute softmax probability distribution and loss.
        """
        num_data, num_feat = x.shape
        _, num_label = self.W.shape
        
        prob = None
        softmax_loss = 0.0
        # ========================= EDIT HERE ========================
        prob = self._softmax(np.dot(x,self.W))

        y_encoded = np.zeros((num_data,num_label))   

        for i in range(num_data) :
            y_encoded[i,y[i]] = 1 

        softmax_loss = -(1/num_data) * np.sum(np.dot(y_encoded,np.transpose(np.log(prob))))

        # ============================================================
        return prob,softmax_loss

    def compute_grad(self, x, y, weight, prob):
        """
        N : # of minibatch data
        D : # of features
        C : # of classes

        Inputs:
        x : (N, D), input data
        weight : (D, C), Weight matrix of classifier
        prob : (N, C), probability distribution over classes for N data
        label : (N, ), label for each data. (0 <= c < C for c in label)

        Returns:
        gradient of weight: (D, C), Gradient of weight to be applied (dL/dW)

        Description:
        Given input (x), weight, probability and label, compute gradient of weight.
        """
        num_data, num_feat = x.shape
        _, num_class = weight.shape

        grad_weight = np.zeros_like(weight, dtype=np.float32)
        # ========================= EDIT HERE ========================
        y_encoded = np.zeros((num_data,num_class))   
        
        for i in range(num_data) :
            y_encoded[i,y[i]] = 1 


        grad_weight = np.dot(np.transpose(x),prob-y_encoded) / num_data
        # ============================================================
        return grad_weight


    def _softmax(self, x):
        """
        Inputs:
        x : (N, C), score before softmax

        Returns:
        softmax : (same shape with x), softmax distribution over axis-1

        Description:
        Given an input x, apply softmax funciton over axis-1.
        """
        softmax = None
        # ========================= EDIT HERE ========================
        softmax = np.exp(x)
        for i in range(x.shape[0]) :
            softmax[i,:] = softmax[i,:] / softmax[i,:].sum()
        # ============================================================
        return softmax
    
    def eval(self, x):
        """

        Inputs:
        x : (N, D), input data

        Returns:
        pred : (N, ), predicted label for N test data

        Description:
        Given N test data, compute probability and make predictions for each data.
        """
        pred, prob = None, None
        # ========================= EDIT HERE ========================
        pred = np.zeros((x.shape[0]))
        prob = self._softmax(np.dot(x,self.W))

        for i in range(x.shape[0]) :
            pred[i] = np.argmax(prob[i])
        
        
        

        # ============================================================
        return pred, prob