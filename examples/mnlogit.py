import nestedlogit
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sms

# ns = 10000
# nc = 5
# nf = 10
ns = 500
nc = 3
nf = 4
X, y = make_classification(n_samples=ns, n_features=nf,
                           n_informative=3, n_redundant=0,
                           n_classes=nc, random_state=2)
# X, y = make_classification(n_samples=40, n_features=3, n_informative=3, n_redundant=0, n_classes=nc, random_state=1)
# print(X)
# print(y)
model = LogisticRegression(
    multi_class='multinomial', penalty=None, fit_intercept=False)
model.fit(X, y)
print("model.coef_:")
print(model.coef_)
print("normalized model.coef_:")
for i in range(1, nc):
    print(model.coef_[i] - model.coef_[0])
model = sms.MNLogit(y, X)
res = model.fit()
print("res.params:")
print(res.params.T)
print(res.summary())

yy = np.zeros((len(y), nc))
yy[np.arange(len(y)), y] = 1.
# res = logitthing.solve("Integer print_level 0\nInteger max_iter 10\nString derivative_test second-order\n", list(range(1, nc)), t, yy, XX, n)
nmodel = nestedlogit.NestedLogitModel(
    yy, X, classes={i: 'y' + str(i + 1) for i in range(nc)},
    nests=list(range(1, nc)),
    availability_vars={i: None for i in range(nc)},
    varz={'x' + str(j + 1): list(range(1, nc)) for j in range(nf)},
    params={nf * i + j: {'x' + str(j + 1): [i]}
            for i in range(1, nc) for j in range(nf)})
res = nmodel.fit(np.zeros(len(nmodel.params)),
                 options={"ipopt": {"print_level": 4}})

# print(nmodel.generate_endog(np.random.default_rng(1), res.params))

print("nestedlogit.NestedLogitModel:")
print(res.summary())

quit()

# autopep8: off

print("With repeating observations:")
# rng = np.random.default_rng(1)
rng = np.random.Generator(np.random.MT19937(1))
nrep = rng.integers(low=50, high=150, size=X.shape[0])
model.fit(np.repeat(X, nrep, axis=0), np.repeat(y, nrep, axis=0))
print("model.coef_:")
print(model.coef_)
print("normalized model.coef_:")
for i in range(1, nc):
    print(model.coef_[i] - model.coef_[0])

res = logitthing.solve("Integer print_level 0\nInteger max_iter 10\nString derivative_test second-order\n", list(range(1, nc)), t, yy, XX, n * np.repeat(nrep, nc))
print("logitthing.solve:")
print(res)

