import math
import random
import numpy as np

from kde import KDEMultivariate
import configspace as cs


class BOHB:
    def __init__(self, configspace, evaluate, resource, eta=3):
        self.eta = eta
        self.configspace = configspace
        self.resource = resource
        self.evaluate = evaluate

        # TODO: Parameter
        self.best_percent = 0.15
        self.random_percent = 1/3
        self.n_samples = 64

        self.s_max = int(math.log(self.resource, self.eta))
        self.budget = (self.s_max + 1) * self.resource

        self.kde_good = None
        self.kde_bad = None
        self.samples = np.array([])
    
    def optimize(self):
        for s in reversed(range(self.s_max + 1)):
            n = int(math.ceil(
                (self.budget * (self.eta ** s)) / (self.resource * (s + 1))))
            r = self.resource * (self.eta ** -s)
            for i in range(s+1):
                n_i = n * self.eta ** ( -i ) # Number of configs
                r_i = r * self.eta ** ( i ) # Budget
                samples = []
                losses = []
                for j in range(n):
                    sample = self.get_sample()
                    loss = self.evaluate(sample.to_dict(), r_i)
                    samples.append(sample)
                    losses.append(loss)

                idxs = np.argsort(losses)[:int(n_i//self.eta)]
                self.samples = np.array(samples)[idxs]

                idxs = np.argsort(losses)
                n_good = int(math.ceil(self.best_percent * len(samples)))
                if n_good > len(self.configspace) + 2:
                    good_data = np.array(samples)[idxs[:n_good]]
                    bad_data = np.array(samples)[idxs[n_good:]]
                    self.good_kde = KDEMultivariate(
                        good_data, self.configspace.kde_vartypes)
                    self.bad_kde = KDEMultivariate(
                        bad_data, self.configspace.kde_vartypes)

    def get_sample(self):
        if self.kde_good is None or random.random() < self.random_percent:
            if self.samples.size:
                # Return successfull sample from the previous budget
                return np.random.choice(self.samples)
            else:
                # Return random configuration as there is no sample to choose from
                return self.configspace.sample_configuration()

        # TODO: Sample from the good data
        return self.configspace.sample_configuration()

hidden_size = cs.CategoricalHyperparameter('hidden_size', [20, 40, 60, 80, 100])
shift = cs.CategoricalHyperparameter('shift', [3, 6, 10, 12, 16])
batch_size = cs.CategoricalHyperparameter('batch_size', [10, 15, 20])
configspace = cs.ConfigurationSpace([hidden_size, shift, batch_size], seed=123123)
bohb = BOHB(configspace, lambda x, t: random.random(), 200)
bohb.optimize()
