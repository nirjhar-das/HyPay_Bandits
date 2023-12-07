import numpy as np
import json
import os
from copy import deepcopy

class HybridBandits:
    def __init__(self, name=None, config=None, load=None):
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
            self.easy = config['is_easy']
            self.num_context = config['num_context']
            self.arms = [self.create_arms(self.easy) for _ in range(self.num_context)]
            self.T = config['horizon_length']
            self.t = 0
            self.context_seq = self.rng.integers(self.num_context, size=self.T)
            self.best_arm, self.max_reward = self.get_best_arm()
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
        theta_proxy = self.rng.standard_normal(size=self.d)
        params['theta'] = self.S1 * theta_proxy / np.linalg.norm(theta_proxy)
        params['beta'] = []
        for i in range(self.L):
            beta_i_proxy = self.rng.standard_normal(size=self.k+1)
            params['beta'].append(self.S2 * beta_i_proxy[:-1] / np.linalg.norm(beta_i_proxy))
        return params
    
    def create_arms(self, easy=False):
        arms = []
        i = 0
        if easy:
            arms.append((self.M*self.parameters['theta']/np.linalg.norm(self.parameters['theta']),\
                         self.N*self.parameters['beta'][0]/np.linalg.norm(self.parameters['beta'][0])))
            while(i < self.L - 1):
                x_proxy = self.rng.standard_normal(size=self.d + 1)
                z_proxy = self.rng.standard_normal(size=self.k + 1)
                x_i = self.N * x_proxy[:-1] / np.linalg.norm(x_proxy)
                z_i = self.M * z_proxy[:-1] / np.linalg.norm(z_proxy)
                reward = np.dot(x_i, self.parameters['theta']) + np.dot(z_i, self.parameters['beta'][i])
                if (reward > 1e-5) and \
                    (reward < (1 - 0.5)*(self.M*np.linalg.norm(self.parameters['theta']) \
                            + self.N*np.linalg.norm(self.parameters['beta'][0]))):
                    arms.append((x_i, z_i))
                    i += 1
        else:
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
        return self.arms[self.context_seq[0]]

    def get_max_reward(self):
        return self.max_reward[self.context_seq[self.t % self.T]]


    def step(self, action):
        if self.model_type == 'Linear':
            noise = self.rng.normal(scale=0.1)
            rewards = [np.dot(self.parameters['theta'], self.arms[self.context_seq[self.t]][a][0]) + \
                        np.dot(self.parameters['beta'][a], self.arms[self.context_seq[self.t]][a][1]) \
                        for a in action]
            noisy_rewards = [reward + noise for reward in rewards]
            regrets = [self.max_reward[self.context_seq[self.t]] - reward for reward in rewards]
            self.t += 1
            return noisy_rewards, regrets, self.arms[self.context_seq[self.t % self.T]]
        elif self.model_type == 'Logistic':
            dot_products = [np.dot(self.parameters['theta'], self.arms[self.context_seq[self.t]][a][0]) + \
                        np.dot(self.parameters['beta'][a], self.arms[self.context_seq[self.t]][a][1]) \
                        for a in action]
            rewards = [1.0 / (1.0 + np.exp(- dot_product)) for dot_product in dot_products]
            noisy_rewards = [float(self.rng.binomial(1, reward)) for reward in rewards]
            regrets = [self.max_reward[self.context_seq[self.t]] - reward for reward in rewards]
            self.t += 1
            return noisy_rewards, regrets, self.arms[self.context_seq[self.t % self.T]]
    
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
