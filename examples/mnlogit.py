import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sms

import nestedlogit

ns = 500
nc = 3
nf = 4
X, y = make_classification(
    n_samples=ns,
    n_features=nf,
    n_informative=3,
    n_redundant=0,
    n_classes=nc,
    random_state=2,
)
model = LogisticRegression(
    multi_class="multinomial", penalty=None, fit_intercept=False
)
model.fit(X, y)
print("model.coef_:")
print(model.coef_)
print("normalized model.coef_:")
for i in range(1, nc):
    print(model.coef_[i] - model.coef_[0])
model = sms.MNLogit(y, X)
res = model.fit()
print("statsmodels.api.MNLogit:")
print(res.summary())

yy = np.zeros((len(y), nc))
yy[np.arange(len(y)), y] = 1.0
casadi_function_opts = {
    "jit": True,
    "compiler": "shell",
    "jit_options": {"compiler": "gcc", "flags": ["-O3"]},
}
nmodel = nestedlogit.NestedLogitModel(
    nestedlogit.NdarrayModelData(yy, X),
    classes={i: "y" + str(i + 1) for i in range(nc)},
    coefficients={
        str(i) + "_" + str(j): {"x" + str(j + 1): [i]}
        for i in range(1, nc)
        for j in range(nf)
    },
    casadi_function_opts=casadi_function_opts,
)
res = nmodel.fit(ipopt_options={"print_level": 2})

print("nestedlogit.NestedLogitModel:")
print(res.summary())
