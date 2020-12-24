import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
        Simple linear regression.
    """
    def __init__(self):
      pass

    def plot(self, xlabel='xlabel', ylabel='ylabel', title='title'):
      plt.figure(figsize=(20,10))
      scores = round(np.mean(self.cv(self.X, self.y)))
      colours = ["#348ABD", "#A60628"]
      plot_x = [min(self.X), max(self.X)]
      plot_y = [self.coef_[0] * x + self.coef_[1] for x in plot_x ]
      plt.scatter(self.X, self.y, label='Data', color=colours[0])
      plt.plot(plot_x, plot_y, label='Model, RMSE: ' + str(scores), color=colours[1])
      plt.legend()
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.title('Linear Regression {} * x + {} '.format(self.k, self.b))
      plt.grid()
      plt.show()

    def coef_(self):
        return self.k[0], self.b[0]

    def predict(self, x):
        list = [ self.coef_[0] * x + self.coef_[1] for x in x ]
        return list

    def rmse(self, predictions, targets):
        return np.sqrt((np.mean((predictions-targets)**2)))

    def cv(self, X, y, cv=2):
        batch = len(X)/cv
        start = 0
        score = []
        for i in range(cv):
            end = int((i+1)*batch)
            predictions = self.predict(X[start:end])
            targets = y[start:end]
            start = int(start + batch)
            score.append(self.rmse(predictions, targets)) 
        return score

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError('X ({}) and y ({}) must have the same length'.format(len(X), len(y)))

        self.X, self.y = X, y
        k_numerator = k_denominator = 0

        for num in range(len(X)):
            k_numerator += (self.X[num] - np.mean(self.X))*(self.y[num] - np.mean(self.y))
            k_denominator += (self.X[num] - np.mean(self.X))**2

        self.k = round((k_numerator / k_denominator)[0])
        self.b = round((np.mean(y) - self.k * np.mean(X)))
      
        self.coef_ = [self.k, self.b]
