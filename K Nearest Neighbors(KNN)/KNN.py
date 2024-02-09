import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class knn:
    
    def __init__(self, k =3):
        self.k = k

    # let's follow the same naming convention has most of the ML libraries like scikit-learn
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y 

    def predict (self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict (self, x):
        #compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        #get the k nearnest samples, labels 
        k_indicies = np.argsort(distances)[:self.k]
        k_neareast_labels =[self.y_train[i] for i in k_indicies]


        #majority vote, we want to get the most comon label 
        most_common = Counter(k_neareast_labels).most_common(1)
        return most_common[0][0]
