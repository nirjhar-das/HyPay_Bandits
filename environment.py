import numpy as np
import json
import os
from copy import deepcopy

class HybridBandits:
    def __init__(self, name=None, config=None, load=None, copy_env=None, seed=None):
        if load is None and copy_env is None:
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
            self.sigma = config['subgaussian']
            self.T = config['horizon_length']
            self.num_context = self.T
            self.arms = [self.create_arms() for _ in range(self.T)]
            if self.model_type == 'Logistic':
                self.kappa = self.calculate_kappa()
            self.t = 0
            self.best_arm, self.max_reward = self.get_best_arm()
        elif copy_env is not None:
            self.name = name
            self.seed = copy_env.seed if seed is None else seed + copy_env.seed
            self.rng = np.random.default_rng(self.seed)
            self.model_type = copy_env.model_type
            self.L = copy_env.L
            self.d = copy_env.d
            self.k = copy_env.k
            self.S1 = copy_env.S1
            self.S2 = copy_env.S2
            self.parameters = copy_env.parameters
            self.M = copy_env.M
            self.N = copy_env.N
            self.sigma = copy_env.sigma
            self.T = copy_env.T
            self.num_context = self.T
            self.arms = copy_env.arms
            if self.model_type == 'Logistic':
                self.kappa = copy_env.kappa
            self.t = 0
            self.best_arm, self.max_reward = copy_env.best_arm, copy_env.max_reward
        else:
            with open(load, 'r') as f:
                data = json.load(f)
                for i in range(data['num_context']):
                    data['arms'][i] = [(np.array(a), np.array(b)) for a,b in data['arms'][i]]
                data['parameters']['theta'] = np.array(data['parameters']['theta'])
                data['parameters']['beta'] = [np.array(beta) for beta in data['parameters']['beta']]
                data['context_seq'] = np.array(data['context_seq'])
                self.__dict__ = deepcopy(data)
                self.rng = np.random.default_rng(self.seed)
    
    
    def create_parameters(self):
        params = {}
        theta_proxy = self.rng.uniform(-1, 1, size=self.d)
        params['theta'] = self.S1 * self.rng.uniform(0, 1) * theta_proxy / np.linalg.norm(theta_proxy)
        params['beta'] = []
        for i in range(self.L):
            beta_i_proxy = self.rng.uniform(-1, 1, size=self.k)
            params['beta'].append(self.S2 * self.rng.uniform(0, 1) * beta_i_proxy / np.linalg.norm(beta_i_proxy))
        return params
    
    def create_arms(self):
        arms = []
        i = 0
        while(i < self.L):
            x_proxy = self.rng.uniform(-1, 1, size=self.d)
            z_proxy = self.rng.uniform(-1, 1, size=self.k)
            x_i = self.rng.uniform(0, self.M) * x_proxy / np.linalg.norm(x_proxy)
            z_i = self.rng.uniform(0, self.N) * z_proxy / np.linalg.norm(z_proxy)
            arms.append((x_i, z_i))
            i += 1
        return arms
    
    def calculate_kappa(self):
        min_mu_dot = np.inf
        for arm_set in self.arms:
            for i, (x,z) in enumerate(arm_set):
                mu_val = 1.0/(1.0 + np.exp(-np.dot(x, self.parameters['theta']) \
                                           - np.dot(z, self.parameters['beta'][i])))
                mu_dot = mu_val * (1.0 - mu_val)
                if mu_dot < min_mu_dot:
                    min_mu_dot = mu_dot
        return 1.0/min_mu_dot
    
    def get_best_arm(self):
        max_reward = [- np.inf for _ in range(self.num_context)]
        best_arm = [-1 for _ in range(self.num_context)]
        for j in range(self.num_context):
            for i in range(self.L):
                if self.model_type == 'Linear':
                    reward = np.dot(self.parameters['theta'], self.arms[j][i][0]) + \
                            np.dot(self.parameters['beta'][i], self.arms[j][i][1])
                elif self.model_type == 'Logistic':
                    reward = 1.0 / (1.0 + np.exp(- np.dot(self.parameters['theta'], self.arms[j][i][0]) \
                            - np.dot(self.parameters['beta'][i], self.arms[j][i][1])))
                if reward > max_reward[j]:
                    best_arm[j] = i
                    max_reward[j] = reward
        return best_arm, max_reward
    
    def get_first_action_set(self):
        return self.arms[0]

    def get_max_reward(self):
        return self.max_reward[self.t]


    def step(self, action):
        if self.model_type == 'Linear':
            noise = self.rng.normal(scale=self.sigma)
            rewards = [np.dot(self.parameters['theta'], self.arms[self.t][a][0]) + \
                        np.dot(self.parameters['beta'][a], self.arms[self.t][a][1]) \
                        for a in action]
            noisy_rewards = [reward + noise for reward in rewards]
            regrets = [self.max_reward[self.t] - reward for reward in rewards]
            self.t += 1
            return noisy_rewards, regrets, self.arms[self.t % self.T]
        else:
            dot_products = [np.dot(self.parameters['theta'], self.arms[self.t][a][0]) + \
                        np.dot(self.parameters['beta'][a], self.arms[self.t][a][1]) \
                        for a in action]
            if self.model_type == 'Logistic':
                rewards = [1.0 / (1.0 + np.exp(- dot_product)) for dot_product in dot_products]
                noisy_rewards = [float(self.rng.binomial(1, reward)) for reward in rewards]
                regrets = [self.max_reward[self.t] - reward for reward in rewards]
                self.t += 1
                return noisy_rewards, regrets, self.arms[self.t % self.T]
    
    def reset(self):
        self.t = 0
    

    def save_metadata(self, folder):
        dict_copy = deepcopy(self.__dict__)
        for i in range(dict_copy['num_context']):
            dict_copy['arms'][i] = [(a.tolist(), b.tolist()) for a, b in dict_copy['arms'][i]]
        dict_copy['parameters']['theta'] = dict_copy['parameters']['theta'].tolist()
        dict_copy['parameters']['beta'] = [beta.tolist() for beta in dict_copy['parameters']['beta']]
        dict_copy['context_seq'] = dict_copy['context_seq'].tolist()
        del dict_copy['rng']
        with open(os.path.join(folder, 'Env_Metadata.json'), 'w+') as f:
            json.dump(dict_copy, f, indent=4)
