import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Make a linear dataset
X, y = make_regression(n_samples=100, n_features=1, noise=0.0)

# Plotting a scatter plot
plt.scatter(x=X, y=y, label='Generated Points')
plt.show()