import numpy as np
import pandas as pd

import nestedlogit

n_samples = 100000

bitgen = np.random.default_rng(seed=1)

exog = pd.DataFrame({'price_a': bitgen.uniform(1., 100., n_samples),
                     'price_b': bitgen.uniform(50., 200., n_samples),
                     'price_c': bitgen.uniform(1., 20., n_samples),
                     'nop_available': np.ones(n_samples)})
endog = pd.DataFrame(np.broadcast_to(np.zeros(1), (n_samples, 4)),
                     columns=['none_count', 'a_count', 'b_count', 'c_count'])

classes = ['a', 'b', 'c']

casadi_function_opts = {
    'jit': True, 'compiler': 'shell',
    'jit_options': {'compiler': 'gcc', 'flags': ['-O3']}}


def init_model(endog, exog, vary_price_sens=False, include_intercept_a=False):
    return nestedlogit.NestedLogitModel(
        nestedlogit.PandasModelData(endog, exog, 10000),
        classes={0: 'none_count', 'a': 'a_count',
                 'b': 'b_count', 'c': 'c_count'},
        nests=['c', ['a', 'b']],
        availability_vars={0: 'nop_available',
                           'a': None, 'b': None, 'c': None},
        params={
            **({'intercept_a': {None: ['a']}} if include_intercept_a else {}),
            'intercept_b': {None: ['b']},
            'intercept_c': {None: ['c']},
            **(
                {'price_sensitivity_' + c: {'price_' + c: [c]}
                 for c in classes}
                if vary_price_sens else
                {'price_sensitivity':
                 {'price_a': ['a'], 'price_b': ['b'], 'price_c': ['c']}}
            )},
        casadi_function_opts=casadi_function_opts)


model = init_model(endog, exog,
                   vary_price_sens=False, include_intercept_a=True)
endog_arr = model.generate_endog(
    bitgen, [1., 2., 3., .01, .6], exog,
    total_counts=bitgen.integers(200, 400, n_samples))
endog = pd.DataFrame(endog_arr, columns=model.endog_out_colnames)

# --- results

print("nop available:")

model = init_model(endog, exog,
                   vary_price_sens=False, include_intercept_a=True)
res = model.fit(ipopt_options={'print_level': 5})
print(res.summary())

model = init_model(endog, exog,
                   vary_price_sens=True, include_intercept_a=True)
res = model.fit(ipopt_options={'print_level': 5})
print(res.summary())

exog['nop_available'][:] = 0
print("nop not available")

model = init_model(endog, exog,
                   vary_price_sens=False, include_intercept_a=False)
res = model.fit(ipopt_options={'print_level': 5})
print(res.summary())

model = init_model(endog, exog,
                   vary_price_sens=True, include_intercept_a=False)
res = model.fit(ipopt_options={'print_level': 5})
print(res.summary())

