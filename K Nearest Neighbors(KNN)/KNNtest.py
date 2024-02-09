import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00', '#0000FF'])

iris = datasets.load_iris()
X,y = iris.data, iris.target
# print(X.shape)
# print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)
# this a 4D array meaning having 4 features having 30 rows meaning observations
# print(X_test.shape)
# print(X_test)
# # of course the y is  1D array meaning having 1 features(result or target feature) having 30 rows meaning observations
# print(y_test.shape)
# print(y_train)

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=y, cmap= cmap, edgecolors='k', s=10)
# plt.show()

from KNN import knn

clf = knn(k=3)
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)


acc = np.sum(prediction == y_test)/ len(y_test)
print(acc)


