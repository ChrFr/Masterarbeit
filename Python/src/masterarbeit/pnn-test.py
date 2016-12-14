import numpy as np

from sklearn import datasets
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from neupy import algorithms, environment

environment.reproducible()

dataset = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.7
)

nw = algorithms.PNN(std=10, verbose=False)
nw.train(x_train, y_train)
result = nw.predict(x_test)
print(metrics.accuracy_score(y_test, result))