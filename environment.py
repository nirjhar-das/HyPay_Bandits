import numpy as np
import json
from copy import deepcopy

class HybridBandits:
    def __init__(self, name, config, load=None):
        if load is None:
            self.name = name
            self.seed = config['seed']
            self.rng = np.random.default_rng(config['seed'])
            self.model_type = config['model_type']
            self.L = config['num_labels']
            self.d = config['theta_dim']
            self.k = config['beta_dim']
            self.S1 = config['theta_norm']
            self.S2 = config['beta_norm']
            self.parameters = self.create_parameters()
            self.M = config['x_norm']
            self.N = config['z_norm']
            self.arms = self.create_arms()
            self.best_arm, self.max_reward = self.get_best_arm()
        else:
            with open(load, 'r') as f:
                data = json.load(f)
                self.__dict__ = deepcopy(data)
                self.rng = np.random.default_rng(self.seed)
    
    
    def create_parameters(self):
        params = {}
        theta_proxy = self.rng.standard_normal(size=self.d+1)
        params['theta'] = self.S1 * theta_proxy[:-1] / np.linalg.norm(theta_proxy)
        params['beta'] = []
        for i in range(self.L):
            beta_i_proxy = self.rng.standard_normal(size=self.k+1)
            params['beta'].append(self.S2 * beta_i_proxy[:-1] / np.linalg.norm(beta_i_proxy))
        return params
    
    def create_arms(self):
        arms = []
        i = 0
        while(i < self.L):
            x_proxy = self.rng.standard_normal(size=self.d + 1)
            z_proxy = self.rng.standard_normal(size=self.k + 1)
            x_i = self.N * x_proxy[:-1] / np.linalg.norm(x_proxy)
            z_i = self.M * z_proxy[:-1] / np.linalg.norm(z_proxy)
            if np.dot(x_i, self.parameters['theta']) + np.dot(z_i, self.parameters['beta'][i]) > 1e-5:
                arms.append((x_i, z_i))
                i += 1
        return arms
    
    def get_best_arm(self):
        max_reward = - np.inf
        best_arm = -1
        for i in range(self.L):
            if self.model_type == 'Linear':
                reward = np.dot(self.parameters['theta'], self.arms[i][0]) + \
                        np.dot(self.parameters['beta'][i], self.arms[i][1])
            elif self.model_type == 'Logistic':
                reward = 1.0 / (1.0 + np.exp(- np.dot(self.parameters['theta'], self.arms[i][0]) \
                        - np.dot(self.parameters['beta'][i], self.arms[i][1])))
            if reward > max_reward:
                best_arm = i
                max_reward = reward
        return best_arm, max_reward


    def step(self, action):
        if self.model_type == 'Linear':
            noise = self.rng.standard_normal()
            reward = np.dot(self.parameters['theta'], self.arms[action][0]) + \
                        np.dot(self.parameters['beta'][action], self.arms[action][1])
            noisy_reward = reward + noise
            regret = self.max_reward - reward
            return noisy_reward, regret
        elif self.model_type == 'Logistic':
            dot_product = np.dot(self.parameters['theta'], self.arms[action][0]) + \
                        np.dot(self.parameters['beta'][action], self.arms[action][1])
            reward = 1.0 / (1.0 + np.exp(- dot_product))
            noisy_reward = self.rng.binomial(1, reward)
            regret = self.max_reward - reward
            return noisy_reward, regret
    

    def save_metadata(self):
        with open(self.name, 'w+') as f:
            json.dump(self.__dict__, f, skipkeys=True)
