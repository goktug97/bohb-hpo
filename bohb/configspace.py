from enum import Enum
import copy
from abc import ABC, abstractmethod
import numbers
from itertools import count

import numpy as np
import scipy 


class Type(Enum):
    Continuous = 'c'
    Discrete = 'o'


class DuplicateHyperparameter(Exception):
    pass


class Configuration:
    def __init__(self, hyperparameters):
        idxs = np.argsort([x._init_idx for x in hyperparameters])
        hyperparameters = np.array(hyperparameters)[idxs]
        self.hyperparameters = []
        self.hyperparameter_map = {}
        self.max_length = 0
        for hyperparameter in hyperparameters:
            length = len(hyperparameter.name)
            if length > self.max_length:
                self.max_length = length
            if hyperparameter.cond is not None:
                if not hyperparameter.cond.compare(self):
                    continue
            if hyperparameter.name in self.hyperparameter_map:
                raise DuplicateHyperparameter(
                    f'Conflicting Hyperparameter: {hyperparameter.name}')
            self.hyperparameter_map[hyperparameter.name] = hyperparameter
            self.hyperparameters.append(hyperparameter)

    def to_dict(self):
        config = {}
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.dont_pass:
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
    _init_count = count()
    def __init__(self, name, value, cond=None, dont_pass=False):
        self._value = None
        self.name = name
        self.value = value
        self.cond = cond
        self._init_idx = next(Hyperparameter._init_count)
        self.dont_pass = dont_pass

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

    def __eq__(self, other):
        if isinstance(other, Hyperparameter):
            return Condition(
                lambda configs: (configs[self.name].value == other.value))
        else:
            return Condition(
                lambda configs: (configs[self.name].value == other))

    def __lt__(self, other):
        if isinstance(other, numbers.Number):
            return Condition(
                lambda configs: (configs[self.name].value < other))
        elif isinstance(other, Hyperparameter):
            return Condition(
                lambda configs: (configs[self.name].value < other.value))
        else:
            raise NotImplementedError

    def __le__(self, other):
        if isinstance(other, numbers.Number):
            return Condition(
                lambda configs: (configs[self.name].value <= other))
        elif isinstance(other, Hyperparameter):
            return Condition(
                lambda configs: (configs[self.name].value <= other.value))
        else:
            raise NotImplementedError

    def __ne__(self, other):
        if isinstance(other, Hyperparameter):
            return Condition(
                lambda configs: (configs[self.name].value != other.value))
        else:
            return Condition(
                lambda configs: (configs[self.name].value != other))

    def __gt__(self, other):
        if isinstance(other, numbers.Number):
            return Condition(
                lambda configs: (configs[self.name].value > other))
        elif isinstance(other, Hyperparameter):
            return Condition(
                lambda configs: (configs[self.name].value > other.value))
        else:
            raise NotImplementedError

    def __ge__(self, other):
        if isinstance(other, numbers.Number):
            return Condition(
                lambda configs: (configs[self.name].value >= other))
        elif isinstance(other, Hyperparameter):
            return Condition(
                lambda configs: (configs[self.name].value >= other.value))
        else:
            raise NotImplementedError


class ConfigurationSpace:
    def __init__(self, hyperparameters, seed=None):
        self.hyperparameters = hyperparameters
        self.rng = np.random.default_rng(seed)
        self.kde_vartypes = ''
        for hyperparameter in self.hyperparameters:
            self.kde_vartypes += hyperparameter.vartype

    def sample_configuration(self):
        hyperparameters = []
        for hyperparameter in self.hyperparameters:
            hyperparameters.append(hyperparameter.sample(self.rng))
        return Configuration(hyperparameters)

    def __len__(self):
        return len(self.hyperparameters)


class Condition:
    def __init__(self, comp):
        self.comp = comp

    def compare(self, configuration):
        return self.comp(configuration.hyperparameter_map)

    def __and__(self, other):
        return Condition(lambda configs: self.comp(configs) and other.comp(configs))

    def __or__(self, other):
        return Condition(lambda configs: self.comp(configs) or other.comp(configs))


class UniformHyperparameter(Hyperparameter):
    def __init__(self, name, lower, upper, cond=None, log=False, dont_pass=False):
        self.type = Type.Continuous
        self._lower = lower
        self._upper = upper
        self.lower = np.log(lower) if log else lower
        self.upper = np.log(upper) if log else upper
        self.log = log
        value = (self.lower + self.upper) / 2
        super().__init__(name, np.exp(value) if log else value, cond, dont_pass)

    def sample(self, rng):
        value = rng.uniform(self.lower, self.upper)
        return self.new(np.exp(value) if self.log else value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = min(max(self._lower, value), self._upper)


class IntegerUniformHyperparameter(UniformHyperparameter):
    def __init__(self, name, lower, upper, cond=None, log=False, dont_pass=False):
        super().__init__(name, lower, upper, cond, log, dont_pass)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = int(min(max(self._lower, value), self._upper))


class ConstantHyperparameter(Hyperparameter):
    def __init__(self, name, value, cond=None):
        self.type = Type.Continuous
        super().__init__(name, value, cond)

    def sample(self, _):
        return self.new(self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class CategoricalHyperparameter(Hyperparameter):
    def __init__(self, name, choices, cond=None, dont_pass=False):
        self.type = Type.Discrete
        self.index = 0
        self.choices = choices
        super().__init__(name, self.index, cond, dont_pass)

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
