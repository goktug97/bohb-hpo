import math
import random
import copy

import numpy as np
import scipy

from bohb.kde import KDEMultivariate
import bohb.configspace as cs


class BOHB:
    def __init__(self, configspace, evaluate, max_budget, min_budget,
                 eta=3, best_percent = 0.15,
                 random_percent = 1/3, n_samples = 64, bw_factor=3, min_bandwidth=1e-3):
        self.eta = eta
        self.configspace = configspace
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.evaluate = evaluate

        self.best_percent = best_percent
        self.random_percent = random_percent
        self.n_samples = n_samples
        self.min_bandwidth = min_bandwidth
        self.bw_factor = bw_factor

        self.s_max = int(math.log(self.max_budget/self.min_budget, self.eta))
        self.budget = (self.s_max + 1) * self.max_budget

        self.kde_good = None
        self.kde_bad = None
        self.samples = np.array([])
    
    def optimize(self):
        min_loss = np.inf
        for s in reversed(range(self.s_max + 1)):
            n = int(math.ceil(
                (self.budget * (self.eta ** s)) / (self.max_budget * (s + 1))))
            r = self.max_budget * (self.eta ** -s)
            self.kde_good = None
            self.kde_bad = None
            self.samples = np.array([])
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
                    if loss < min_loss:
                        min_loss = loss
                        best_hyperparameter = sample

                idxs = np.argsort(losses)[:int(n_i//self.eta)]
                self.samples = np.array(samples)[idxs]

                idxs = np.argsort(losses)
                n_good = int(math.ceil(self.best_percent * len(samples)))
                if n_good > len(self.configspace) + 2:
                    good_data = np.array(samples)[idxs[:n_good]]
                    bad_data = np.array(samples)[idxs[n_good:]]
                    self.kde_good = KDEMultivariate(
                        good_data, self.configspace.kde_vartypes)
                    self.kde_bad = KDEMultivariate(
                        bad_data, self.configspace.kde_vartypes)
                    self.kde_bad.kde.bw = np.clip(
                        self.kde_bad.kde.bw, self.min_bandwidth,None)
                    self.kde_good.kde.bw = np.clip(
                        self.kde_good.kde.bw, self.min_bandwidth,None)
        return best_hyperparameter

    def get_sample(self):
        if self.kde_good is None or random.random() < self.random_percent:
            if self.samples.size:
                # Return successfull sample from the previous budget
                return np.random.choice(self.samples)
            else:
                # Return random configuration as there is no sample to choose from
                return self.configspace.sample_configuration()

        # Sample from the good data
        for _ in range(self.n_samples):
            idx = np.random.randint(0, self.kde_good.data.shape[0])
            best_tpe_val = np.inf
            configuration = copy.deepcopy(self.kde_good.configurations[idx])
            for hyperparameter, bw in zip(configuration, self.kde_good.kde.bw):
                if hyperparameter.value['index'] == -1:
                    value = hyperparameter.value['value']
                    bw = bw * self.bw_factor
                    # https://github.com/automl/HpBandSter/blob/841db4b827f342e5eb7f725723ea6461ac52d45a/hpbandster/optimizers/config_generators/bohb.py#L156
                    hyperparameter.value['value'] = scipy.stats.truncnorm.rvs(
                        -value/bw,(1-value)/bw, loc=value, scale=bw)
                else:
                    if np.random.rand() >= (1-bw):
                        hp = self.configspace.hyperparameter_map[hyperparameter.name]
                        idx = np.random.randint(len(hp.choices))
                        hyperparameter.value['index'] = idx
                        hyperparameter.value['value'] = hp.choices[idx]

            tpe_val = (self.kde_bad.kde.pdf(configuration.to_list())/
                       self.kde_good.kde.pdf(configuration.to_list()))
            if tpe_val < best_tpe_val:
                best_tpe_val = tpe_val
                best_configuration = configuration

        return best_configuration
