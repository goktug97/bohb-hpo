from enum import Enum
import copy
from abc import ABC, abstractmethod

import numpy as np
import scipy 


class Type(Enum):
    Continuous = 'c'
    Discrete = 'o'


class Configuration:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.max_length = max(
            [len(hyperparameter.name)
             for hyperparameter in self.hyperparameters])

    def to_dict(self):
        config = {}
        for hyperparameter in self.hyperparameters:
            config[hyperparameter.name] = hyperparameter.value
        return config

    def to_list(self):
        array = []
        for hyperparameter in self.hyperparameters:
            if hyperparameter.type == Type.Continuous:
                array.append(hyperparameter.value)
            elif hyperparameter.type == Type.Discrete:
                array.append(hyperparameter.index)
            else:
                raise NotImplementedError
        return array

    def __getitem__(self, idx):
        return self.hyperparameters[idx]

    def __str__(self):
        string = ["Configuration:\n"]
        for hyperparameter in self.hyperparameters:
            string.append(
                (f'{"Name:":>8} {hyperparameter.name: <{self.max_length}} | '
                 f"Value: {hyperparameter.value}\n").ljust(10))
        return ''.join(string)


class Hyperparameter(ABC):
    def __init__(self, name, value):
        self._value = None
        self.name = name
        self.value = value

    def new(self, value=None):
        new_hyperparameter = copy.deepcopy(self)
        if value is not None:
            new_hyperparameter.value = value
        return new_hyperparameter

    @abstractmethod
    def sample(self):
        ...

    @property
    @abstractmethod
    def value(self):
        ...

    @value.setter
    @abstractmethod
    def value(self, value):
        ...

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        self.vartype = type.value
        self._type = type


class ConfigurationSpace:
    def __init__(self, hyperparameters, seed=None):
        self.hyperparameters = hyperparameters
        self.rng = np.random.default_rng(seed)
        self.kde_vartypes = ''
        self.hyperparameter_map = {}
        for hyperparameter in self.hyperparameters:
            self.kde_vartypes += hyperparameter.vartype
            self.hyperparameter_map[hyperparameter.name] = hyperparameter

    def sample_configuration(self):
        hyperparameters = []
        for hyperparameter in self.hyperparameters:
            hyperparameters.append(hyperparameter.sample(self.rng))
        return Configuration(hyperparameters)

    def __len__(self):
        return len(self.hyperparameters)


class UniformHyperparameter(Hyperparameter):
    def __init__(self, name, lower, upper, log=False):
        self.type = Type.Continuous
        self.lower = lower
        self.upper = upper
        self.log = log
        super().__init__(name, (lower + upper) / 2)

    def sample(self, rng):
        func = (scipy.stats.loguniform.rvs if self.log
                else rng.uniform)
        value = func(self.lower, self.upper)
        return self.new(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not self.log:
            self._value = min(max(self.lower, value), self.upper)
        else:
            self._value = value


class IntegerUniformHyperparameter(UniformHyperparameter):
    def __init__(self, name, lower, upper, log):
        super().__init__(name, lower, upper, log)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not self.log:
            self._value = int(min(max(self.lower, value), self.upper))
        else:
            self._value = value

    def sample(self, rng):
        # TODO: Log
        func = rng.integers
        value = func(self.lower, self.upper)
        return self.new(value)


class CategoricalHyperparameter(Hyperparameter):
    def __init__(self, name, choices):
        self.type = Type.Discrete
        self.index = 0
        self.choices = choices
        super().__init__(name, self.index)

    def sample(self, rng):
        index = rng.integers(0, len(self.choices))
        return self.new(index)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, index):
        self.index = index
        self._value = self.choices[index]
