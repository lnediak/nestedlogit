import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=4, random_state=1)
X[y == 0] = np.zeros_like(X[y == 0])
model = LogisticRegression(multi_class='multinomial', penalty='none', fit_intercept=False)
model.fit(X, y)
print(model.coef_)

import logitthing
res = logitthing.solve("Integer max_iter 10\nString derivative_test second-order\n", np.arange(len(y)), y, X, np.ones_like(y))
print(res)

