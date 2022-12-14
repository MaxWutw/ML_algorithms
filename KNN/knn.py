## K Nearest Neighbor
import numpy as np
from collections import Counter
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y


    def predict(self, x):
        predicted_labels = [self._predict(i) for i in x]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute euclidean distance
        distance = [euclidean_distance(x, x_train) for x_train in self.x_train]

        # get k nearest samples and labels
        k_indices = np.argsort(distance)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # calculate the most common sample
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]