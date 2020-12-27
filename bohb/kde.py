import numpy as np
import statsmodels.api as sm

class KDEMultivariate:
    def __init__(self, configurations, vartypes):
        self.configurations = configurations
        self.vartypes = vartypes
        self.data = []
        for config in configurations:
            self.data.append(np.array(config.to_list()))
        self.data = np.array(self.data)
        self.kde = sm.nonparametric.KDEMultivariate(
            self.data, vartypes, 'normal_reference')
