# implementation_algorithms

```python
from sklearn.datasets import make_regression
X, y = make_regression(n_features=1, noise=65)
```

```python
lreg = LinearRegression()
lreg.fit(X, y)
```

```python
lreg.coef_
# output: [92, 5]

lreg.predict([-1, 24])
# output: [-87, 2213]
```

```python
lreg.plot()
```

![Linear Regression](lin_reg_plot.png)

