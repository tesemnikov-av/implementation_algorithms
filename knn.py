import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:

    def __init__(self, k=3):
        self.k = k
        self.trained = True

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.trained = True

    def predict(self, X_test):
        self.__assert()
        predictions = []
        for x in X_test:
            distances = [euclidean_distance(x, X_train) for X_train in self.X]
            sort_distances = np.argsort(distances)[:self.k]
            
            labels = []
            for dis in sort_distances:
              labels.append(self.y[dis])
    
            most_frequent = Counter(labels).most_common(1)
            predictions.append(most_frequent[0][0])
        return predictions # self.y[np.argmin(distances)], labels#, np.argsort(distances), distances

    def accuracy(self, X_test, y_test):
        self.__assert()
        pass

    def __assert(self):
        assert self.trained, 'First you need to fit a model'
