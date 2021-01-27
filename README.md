# VRNK - ML with Numpy only

## Linear Regression

```python
from sklearn.datasets import make_regression
from linear_regression import LinearRegression

X, y = make_regression(n_features=1, noise=65)

lreg = LinearRegression()
lreg.fit(X, y)

lreg.coef_
# output: [92, 5]

lreg.predict([-1, 24])
# output: [-87, 2213]

lreg.plot()

# Need Confidence interval

```

![Linear Regressions](https://github.com/tesemnikov-av/implementation_algorithms/blob/main/pics/linreg_plot.png?raw=true)

## K-nearest Neighbors Algorithm (https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("custom KNN classification accuracy", accuracy(y_test, predictions))
# output: custom KNN classification accuracy 0.9666666666666667
```
