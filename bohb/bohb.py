import math
import random
import copy

import numpy as np
import scipy
import statsmodels.api as sm
try:
    import ray
    ray_available = True
except ModuleNotFoundError:
    class ray:
        @staticmethod
        def remote(func):
            return func
    ray_available = False

import bohb.configspace as cs


@ray.remote
def _evaluate(evaluate, r_i, sample):
    loss = evaluate(sample.to_dict(), r_i)
    return loss


class KDEMultivariate(sm.nonparametric.KDEMultivariate):
    def __init__(self, configurations, vartypes):
        self.configurations = configurations
        self.vartypes = vartypes
        data = []
        for config in configurations:
            data.append(np.array(config.to_list()))
        data = np.array(data)
        super().__init__(data, vartypes, 'normal_reference')


class BOHB:
    def __init__(self, configspace, evaluate, max_budget, min_budget,
                 eta=3, best_percent=0.15, random_percent=1/3, n_samples=64,
                 bw_factor=3, min_bandwidth=1e-3, enable_ray=False):
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
        self.enable_ray = enable_ray

        self.s_max = int(math.log(self.max_budget/self.min_budget, self.eta))
        self.budget = (self.s_max + 1) * self.max_budget

        self.kde_good = None
        self.kde_bad = None
        self.samples = np.array([])

        if ray_available and enable_ray:
            ray.init()

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
                n_i = n * self.eta ** (-i)  # Number of configs
                r_i = r * self.eta ** (i)  # Budget

                # TODO: Not working?
                if ray_available and self.enable_ray:
                    samples = [self.get_sample() for _ in range(n)]
                    results = [_evaluate.remote(self.evaluate, r_i, sample)
                               for sample in samples]
                    losses = ray.get(results)
                    loss_idx = np.argmin(losses)
                    if losses[loss_idx] < min_loss:
                        min_loss = losses[loss_idx]
                        best_hyperparameter = samples[loss_idx]
                else:
                    samples = []
                    losses = []
                    for j in range(n):
                        sample = self.get_sample()
                        loss = self.evaluate(sample.to_dict(), int(r_i))
                        samples.append(sample)
                        losses.append(loss)
                        if loss < min_loss:
                            min_loss = loss
                            best_hyperparameter = sample

                n = int(n_i//self.eta)
                idxs = np.argsort(losses)
                self.samples = np.array(samples)[idxs[:n]]
                n_good = int(math.ceil(self.best_percent * len(samples)))
                if n_good > len(self.configspace) + 2:
                    good_data = np.array(samples)[idxs[:n_good]]
                    bad_data = np.array(samples)[idxs[n_good:]]
                    self.kde_good = KDEMultivariate(
                        good_data, self.configspace.kde_vartypes)
                    self.kde_bad = KDEMultivariate(
                        bad_data, self.configspace.kde_vartypes)
                    self.kde_bad.bw = np.clip(
                        self.kde_bad.bw, self.min_bandwidth, None)
                    self.kde_good.bw = np.clip(
                        self.kde_good.bw, self.min_bandwidth, None)
        return best_hyperparameter

    def get_sample(self):
        if self.kde_good is None or np.random.random() < self.random_percent:
            return self.configspace.sample_configuration()

        # Sample from the good data
        best_tpe_val = np.inf
        for _ in range(self.n_samples):
            idx = np.random.randint(0, len(self.kde_good.configurations))
            configuration = copy.deepcopy(self.kde_good.configurations[idx])
            for hyperparameter, bw in zip(configuration, self.kde_good.bw):
                if hyperparameter.type == cs.Type.Continuous:
                    value = hyperparameter.value
                    bw = bw * self.bw_factor
                    hyperparameter.value = scipy.stats.truncnorm.rvs(
                        -value/bw, (1-value)/bw, loc=value, scale=bw)
                elif hyperparameter.type == cs.Type.Discrete:
                    if np.random.rand() >= (1-bw):
                        idx = np.random.randint(len(hyperparameter.choices))
                        hyperparameter.value = idx
                else:
                    raise NotImplementedError

            tpe_val = (self.kde_bad.pdf(configuration.to_list()) /
                       self.kde_good.pdf(configuration.to_list()))
            if tpe_val < best_tpe_val:
                best_tpe_val = tpe_val
                best_configuration = configuration

        return best_configuration
