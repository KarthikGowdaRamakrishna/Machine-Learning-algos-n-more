import numpy as np

class Logisticregression:

    def __init__(self, lr=0.001, n_iters = 1000):
        self.lr =lr
        self.n_iters = n_iters
        self.weights = None
        self.bais =None

    def fit(self, X, y):
        #init the parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bais = 0

        #gradient descent
        for _ in range(self.n_iters):
            #first to compute the sigmoid function we need the linear function to input to it
            #f(w,b) = wx + b
            linear_model = np.dot(X, self.weights) + self.bais
            y_predicted = self._sigmoid(linear_model)
            #calculating derivaties w.r.t weights
            dw = (1/n_samples) * np.dot(X.T, (y_predicted -y))
            #calculating derivaties w.r.t bais
            db = (1/ n_samples) * np.sum(y_predicted - y) 

            self.weights -= self.lr * dw
            self.bais -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bais
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1/(1+ np.exp(-x))