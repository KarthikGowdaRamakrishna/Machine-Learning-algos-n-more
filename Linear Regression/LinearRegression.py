import numpy as np


#f(x) or y = mx + b  
#---> f(x), y=> function of the i/p var, or simply 'y' that's o/p; x=> input var/ independent var
# m or w,beta1,weights => controls the slope of the line(this coeficent variance )
## To calculate m or w =>NEW w = OLD w - alpha.dw 
####alpha = learning rate how stip it should jump down ; dw = derivative of the all the weights  
# b,beta0,bias => controls the intercept(on/start of y axis) of the line (this coeficent bais)
## To calculate b=>NEW w = OLD b - alpha.bw 
####alpha = learning rate how stip it should jump down ; db = derivative of the all the b's  
#beta0,b1,b2,b3..... =>vizulatize the linearReg. as a hyper plain for higher dimnesions  
class LinearRegression:
    def __init__(self, lr=0.01, n_iters=10):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias =None

    def fit(self, X, y):
        #gradient descend method implimentation it needs to start from some point let's....
        #init that parameter
        n_samples, n_features = X.shape
        #init x values from zero of course cause we start from the i/p x=0 but could be any value it will change anyways
        self.weights = np.zeros(n_features)
        #beta2 value that's the y-axis, messing with that will change the bais
        self.bias = 0

        #f(x) or y = mx + b  
        for _ in range(self.n_iters):
                          #this will multiple X with the weights
            y_predicted = np.dot(X, self.weights) + self.bias
            print("X :")
            print(X)
            print("weights :")
            print(self.weights)
            print("y_pred")
            print(y_predicted)
            print("y:")
            print(y)

            #calculating derivative w.t.r 'w'
            ## dj/dw = dw = 1/N(n sum i=1)2xi(y -yi)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted -y))
            print("dw:")
            print(dw)
            db = (1/n_samples) * np.sum(y_predicted-y)
            print("db")
            print(db)

            self.weights -= self.lr * dw
            print("weights")
            print(self.weights)
            self.bias -= self.lr * db
            print("bias")
            print(self.bias)


    def predict(self, X):

        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted