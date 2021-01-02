Bayesian Optimization Hyperband Hyperparameter Optimization
===========================================================

Implementation for [BOHB](http://proceedings.mlr.press/v80/falkner18a.html)

## Requirements
    - numpy
    - scipy
    - statsmodels
    - dask
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


if __name__ == '__main__':
    alpha = cs.CategoricalHyperparameter('alpha', [0.001, 0.01, 0.1])
    beta = cs.CategoricalHyperparameter('beta', [1, 2, 3])
    configspace = cs.ConfigurationSpace([alpha, beta])

    opt = BOHB(configspace, evaluate, max_budget=10, min_budget=1)

    # Parallel
    # opt = BOHB(configspace, evaluate, max_budget=10, min_budget=1, n_proc=4)

    logs = opt.optimize()
```

See [examples](https://github.com/goktug97/bohb-hpo/tree/master/examples)

### Configspace Examples

- Basic
```python
import dehb.configspace as cs
lr = cs.UniformHyperparameter('lr', 1e-4, 1e-1, log=True)
batch_size = cs.CategoricalHyperparameter('batch_size', [8, 16, 32])
configspace = cs.ConfigurationSpace([lr, batch_size], seed=123)
```

- Conditional Parameters
```python
import bohb.configspace as cs
a = cs.IntegerUniformHyperparameter('a', 0, 4)
b = cs.CategoricalHyperparameter('b', ['a', 'b', 'c'], a == 0)
b_default = cs.CategoricalHyperparameter('b', ['d'], ~b.cond)
configspace = cs.ConfigurationSpace([a, b, b_default], seed=123)
```

- Complex Conditional Parameters
```python
import bohb.configspace as cs
a = cs.IntegerUniformHyperparameter('a', 0, 4)
b1 = cs.UniformHyperparameter('b', 0, 0.5, a <= 1)
b2 = cs.UniformHyperparameter('b', 0.5, 1, ~b1.cond)
c1 = cs.CategoricalHyperparameter('c', ['a', 'b', 'c'], b1 < 0.25)
c2 = cs.CategoricalHyperparameter('c', ['c', 'd', 'e'], ~c1.cond)
d1 = cs.UniformHyperparameter('d', 0, 1, (b1 < 0.125) & (c1 == 'b'))
d2 = cs.UniformHyperparameter('d', 0, 0, ~d1.cond)
configspace = cs.ConfigurationSpace([a, b1, b2, c1, c2, d1, d2], seed=123)
```

## TODO
    - More Hyperparameters

## License
bohb-hpo is licensed under the MIT License.
