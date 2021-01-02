Bayesian Optimization Hyperband Hyperparameter Optimization
===========================================================

Implementation for [BOHB](http://proceedings.mlr.press/v80/falkner18a.html)

## Requirements
    - numpy
    - scipy
    - statsmodels
    - torch (example)

## Installation
```bash
pip3 install bohb-hpo
```

## Usage

``` Python
from bohb import BOHB
import bohb.configspace as cs


def objective(step, alpha, beta):
    return 1 / (alpha * step + 0.1) + beta


def evaluate(params, n_iterations):
    loss = 0.0
    for i in range(int(n_iterations)):
        loss += objective(**params, step=i)
    return loss/n_iterations


alpha = cs.CategoricalHyperparameter('alpha', [0.001, 0.01, 0.1])
beta = cs.CategoricalHyperparameter('beta', [1, 2, 3])
configspace = cs.ConfigurationSpace([alpha, beta], seed=123)

opt = BOHB(configspace, evaluate, max_budget=10, min_budget=1)
logs = opt.optimize()
```

See [examples](https://github.com/goktug97/bohb-hpo/tree/master/examples)

## TODO
    - Parallel Optimization (Implemented but not working properly)
    - More Hyperparameters

## License
bohb-hpo is licensed under the MIT License.
