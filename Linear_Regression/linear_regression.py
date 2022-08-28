## Linear Regression
import numpy as np

class LinearRegression:
    def __init__(self, lr=3e-1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        ## Gradient Descent
        for _ in range(self.n_iters):
            predict = np.dot(x, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(x.T, (predict - y)) # dot product
            db = (1/n_samples) * np.sum(predict - y) # dot product
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    
    def predict(self, x):
        predict = self.weights * x + self.bias
        return predict
