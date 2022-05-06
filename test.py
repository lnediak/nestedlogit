import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

nc = 3
#X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=nc, random_state=1)
X, y = make_classification(n_samples=20, n_features=3, n_informative=3, n_redundant=0, n_classes=nc, random_state=1)
print(X)
print(y)
model = LogisticRegression(multi_class='multinomial', penalty='none', fit_intercept=False)
model.fit(X, y)
print(model.coef_)

import logitthing
yy = np.tile(np.arange(nc), len(y))
XX = np.repeat(X, nc, axis=0)
t = np.repeat(np.arange(len(y)), nc)
ytmp = np.repeat(y, nc)
n = np.array(yy == ytmp, dtype='int64')
res = logitthing.solve("Integer print_level 1\nInteger max_iter 10\nString derivative_test second-order\n", t, yy, XX, n)
print(res)

