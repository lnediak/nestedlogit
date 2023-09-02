import numpy as np
import pandas as pd

import nestedlogit

n_samples = 100000

bitgen = np.random.default_rng(seed=1)

exog = pd.DataFrame(
    {
        "price_a": bitgen.uniform(1.0, 100.0, n_samples),
        "price_b": bitgen.uniform(50.0, 200.0, n_samples),
        "price_c": bitgen.uniform(1.0, 20.0, n_samples),
        "nop_available": np.ones(n_samples),
    }
)
classes = ["a", "b", "c"]
all_classes = ["nop"] + classes
endog = pd.DataFrame(
    np.broadcast_to(np.zeros(1), (n_samples, 4)),
    columns=all_classes,
)

casadi_function_opts = {
    "jit": True,
    "compiler": "shell",
    "jit_options": {"compiler": "gcc", "flags": ["-O3"]},
}


def init_model(endog, exog, vary_price_sens=False, include_intercept_a=False):
    return nestedlogit.NestedLogitModel(
        nestedlogit.data.SimpleModelData((endog, exog), max_rows=10000),
        classes=all_classes,
        nests={"nest_ab": ["a", "b"]},
        availability_vars={"nop": "nop_available"},
        coefficients={
            **({"intercept_a": {None: ["a"]}} if include_intercept_a else {}),
            "intercept_b": {None: ["b"]},
            "intercept_c": {None: ["c"]},
            **(
                {
                    "price_sensitivity_" + c: {"price_" + c: [c]}
                    for c in classes
                }
                if vary_price_sens
                else {
                    "price_sensitivity": {
                        "price_a": ["a"],
                        "price_b": ["b"],
                        "price_c": ["c"],
                    }
                }
            ),
        },
        casadi_function_opts=casadi_function_opts,
    )


model = init_model(
    endog, exog, vary_price_sens=False, include_intercept_a=True
)
### data generation is here:
endog = model.generate_endog(
    bitgen,
    {
        "intercept_a": 1.0,
        "intercept_b": 2.0,
        "intercept_c": 3.0,
        "price_sensitivity": 0.01,
        "nest_ab": 0.6,
    },
    exog,
    total_counts=bitgen.integers(200, 400, n_samples),
)

# --- results

print("nop available:")

model = init_model(
    endog, exog, vary_price_sens=False, include_intercept_a=True
)
res = model.fit(ipopt_options={"print_level": 5})
res.use_fit_null(ipopt_options={"print_level": 3})
print(res.summary())

model = init_model(endog, exog, vary_price_sens=True, include_intercept_a=True)
res = model.fit(ipopt_options={"print_level": 5})
res.use_fit_null(ipopt_options={"print_level": 3})
print(res.summary())

exog["nop_available"][:] = 0
print("nop not available")

model = init_model(
    endog, exog, vary_price_sens=False, include_intercept_a=False
)
res = model.fit(ipopt_options={"print_level": 5})
res.use_fit_null(ipopt_options={"print_level": 3})
print(res.summary())

model = init_model(
    endog, exog, vary_price_sens=True, include_intercept_a=False
)
res = model.fit(ipopt_options={"print_level": 5})
res.use_fit_null(ipopt_options={"print_level": 3})
print(res.summary())
