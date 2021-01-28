import numpy as np

class LR:

    def __init__(self, learning_rate=0.1, epochs=10):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = 0
        self.bias = 0

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.bias = 1
        self.weights = np.ones(n_features)

        for epoch in range(self.epochs):
            
            predictons = self.predict(X)
            
            dw = (1 / n_samples) * np.dot(X.T, (predictons - y))
            db = (1 / n_samples) * np.sum(predictons - y)

            self.weights -= self.learning_rate * dw
            self.weights -= self.learning_rate * db

    def predict(self, X):
        
        predictions = np.dot(X, self.weights) + self.bias
        return predictions
