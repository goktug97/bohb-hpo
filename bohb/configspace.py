import numpy as np


class Configuration:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.max_length = max(
            [len(hyperparameter.name) for hyperparameter in self.hyperparameters])

    def to_dict(self):
        config = {}
        for hyperparameter in self.hyperparameters:
            config[hyperparameter.name] = hyperparameter.value['value']
        return config

    def to_list(self):
        array = []
        for hyperparameter in self.hyperparameters:
            if hyperparameter.value['index'] == -1:
                array.append(hyperparameter.value['value'])
            else:
                array.append(hyperparameter.value['index'])
        return array

    def __str__(self):
        string = ["Configuration:\n"]
        for hyperparameter in self.hyperparameters:
            string.append((f'{"Name:":>8} {hyperparameter.name: <{self.max_length}} | '
                           f"Value: {hyperparameter.value['value']}\n").ljust(10))
        return ''.join(string)


class Hyperparameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class ConfigurationSpace:
    def __init__(self, hyperparameters, seed=None):
        self.hyperparameters = hyperparameters
        self.rng = np.random.default_rng(seed)
        self.kde_vartypes = ''.join(
            [hyperparameter.vartype for hyperparameter in self.hyperparameters])

    def sample_configuration(self):
        hyperparameters = []
        for hyperparameter in self.hyperparameters:
            value = hyperparameter.sample(self.rng)
            hyperparameters.append(
                Hyperparameter(hyperparameter.name, value))
        return Configuration(hyperparameters)

    def __len__(self):
        return len(self.hyperparameters)


class UniformHyperparameter(Hyperparameter):
    def __init__(self, name, lower, upper, integer=False):
        value = (lower + upper) / 2
        value = int(value) if integer else value
        super().__init__(name, value)
        self.lower = lower
        self.upper = upper
        self.integer = integer
        self.vartype = 'c'

    def sample(self, rng):
        func = rng.integers if self.integer else rng.uniform
        value = func(self.lower, self.upper)
        return {'index': -1, 'value': value}


class CategoricalHyperparameter(Hyperparameter):
    def __init__(self, name, choices):
        super().__init__(name, choices[0])
        self.choices = choices
        self.vartype = 'o'

    def sample(self, rng):
        index = rng.integers(0, len(self.choices))
        return {'index': index, 'value': self.choices[index]}
