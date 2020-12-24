# implementation_algorithms

```python
from sklearn.datasets import make_regression
X, y = make_regression(n_features=1, noise=65)
```

```python
lreg = LinearRegression()
lreg.fit(X, y)

lreg.plot()
```

