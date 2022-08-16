import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from knn import KNN
iris = datasets.load_iris() ## https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
## iris 150 * 4 ndarray, it construct with sepal and petal's length and width.

x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=456)

cmap = matplotlib.colors.ListedColormap(['#FF0000', '#1E90FF', '#006400'])

clf = KNN(k=3)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
# print(x_test)
accuracy = np.sum(prediction == y_test) / len(y_test)
print(accuracy)

plt.figure(1,figsize = (13,6))
plt.subplot(121)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap, edgecolor='k', s=20)
plt.xlabel("Origin")

new_x, new_y = np.append(x_train, x_test, axis=0), np.append(y_train, prediction, axis=0)
# print(new_x)
# print(new_y)
# print(y_train)

plt.subplot(122)
plt.scatter(new_x[:, 0], new_x[:, 1], c=new_y, cmap=cmap, edgecolor='k', s=20)
plt.xlabel("KNN")

plt.show()