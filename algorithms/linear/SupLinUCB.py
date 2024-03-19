from ..algorithm import Algorithm
from copy import deepcopy
import numpy as np
import scipy as sc
import pandas as pd

class SupLinUCB(Algorithm):
    def __init__(self, arms, delta, M, N, S1, S2, sigma, lmbda, T, info=None):
        super().__init__(f'SupLinUCB_{info}' if info is not None else 'SupLinUCB', arms)
        self.M = np.sqrt(M*M + N*N)
        self.S = np.sqrt(S1*S1 + self.L*S2*S2)
        self.lmbda = lmbda
        self.delta = delta
        self.sigma = sigma
        self.T = T
        self.num_levels = int(np.ceil(np.log(self.T)))
        self.levels_dict = [{'V_tilde': self.lmbda * np.eye(self.d),
                             'V_tilde_inv': (1 / self.lmbda) * np.eye(self.d),
                         'theta_hat': np.zeros((self.d,)),
                         'beta_hat_arr': [np.zeros((self.k,)) for _ in range(self.L)],
                         'W_arr': [self.lmbda*np.eye(self.k) for _ in range(self.L)],
                         'W_arr_inv': [(1 / self.lmbda)*np.eye(self.k) for _ in range(self.L)],
                         'v_arr': [np.zeros((self.k,)) for _ in range(self.L)],
                         'B_arr': [np.zeros((self.d, self.k)) for _ in range(self.L)],
                         'u': np.zeros((self.d,))} for _ in range(self.num_levels)]
        self.t = 0
        self.a_t = 0
        self.level_to_be_updated = -1
        self.update_level_flag = False
    
    def get_estimates_and_confidence(self, level, arm_idx):
        d = self.levels_dict[level]
        arm_estimates = {i: [] for i in arm_idx}
        conf_radius = (np.sqrt(self.lmbda)*self.S + np.sqrt(2*np.log(self.T*self.L/self.delta)))
        for idx in arm_idx:
            mean = np.dot(self.arms[idx][0], d['theta_hat']) + np.dot(self.arms[idx][1], d['beta_hat_arr'][idx])
            bonus = np.sqrt(np.dot(self.arms[idx][0], np.dot(d['V_tilde_inv'], self.arms[idx][0])) -\
                2*np.dot(self.arms[idx][0], np.dot(d['V_tilde_inv'] @ d['B_arr'][idx] @ d['W_arr_inv'][idx], self.arms[idx][1])) +\
                np.dot(self.arms[idx][1], np.dot(d['W_arr_inv'][idx], self.arms[idx][1])) +\
                np.dot(self.arms[idx][1], np.dot(d['W_arr_inv'][idx] @ d['B_arr'][idx].T @ d['V_tilde_inv'] @ d['B_arr'][idx] @ d['W_arr_inv'][idx], self.arms[idx][1])))
            arm_estimates[idx] += [mean + conf_radius * bonus, conf_radius * bonus]
        
        return arm_estimates
    
    def next_action(self):
        s = 0
        arm_idx = [i for i in range(self.L)]
        while(True):
            arm_estimates = self.get_estimates_and_confidence(s, arm_idx)
            # print(arm_idx)
            # print(arm_estimates)
            bonus_arr = np.array([arm_estimates[i][1] for i in arm_estimates.keys()])
            if np.prod(bonus_arr <= (1/np.sqrt(self.T))) == 1:
                print('First Cond:', s)
                self.a_t = int(max(arm_estimates, key = lambda x: arm_estimates.get(x)[0]))
                return self.a_t
            elif np.prod(bonus_arr <= (1.0/np.power(2.0, s+1))) == 1:
                a_max = int(max(arm_estimates, key = lambda x: arm_estimates.get(x)[0]))
                mask = np.array([arm_estimates[i][0] for i in arm_estimates.keys()]) >= (arm_estimates[a_max][0] - (1.0/np.power(2.0, s)))
                arm_idx = np.array(arm_idx)[mask].tolist()
                s += 1
            else:
                mask = bonus_arr > (1/np.power(2.0, s+1))
                valid_idx = np.array(arm_idx)[mask].tolist()
                self.a_t = valid_idx[0]
                self.level_to_be_updated = s
                self.update_level_flag = True
                return self.a_t
    

    def update(self, reward, regret, arm_set):
        if self.update_level_flag:
            d = deepcopy(self.levels_dict[self.level_to_be_updated])
            x = self.arms[self.a_t][0]
            z = self.arms[self.a_t][1]
            V_tilde = d['V_tilde']
            W = d['W_arr'][self.a_t]
            W_inv = d['W_arr_inv'][self.a_t]
            B = d['B_arr'][self.a_t]
            u = d['u']
            v = d['v_arr'][self.a_t]
            u += np.dot(B @ W_inv, v)
            V_tilde = V_tilde + B @ W_inv @ B.T
            B += np.outer(x, z)
            W += np.outer(z, z)
            W_inv = np.linalg.inv(W)
            v += reward * z
            V_tilde += (np.outer(x, x) - \
                            B @ W_inv @ B.T)
            u += reward * x - \
                        np.dot(B @ W_inv, v)
            V_tilde_inv = np.linalg.inv(V_tilde)
            d['theta_hat'] = np.dot(V_tilde_inv, u)
            d['V_tilde'] = V_tilde
            d['W_arr'][self.a_t] = W
            d['W_arr_inv'][self.a_t] = W_inv
            d['B_arr'][self.a_t] = B
            d['u'] = u
            d['v_arr'][self.a_t] = v
            for i in range(self.L):
                d['beta_hat_arr'][i] = np.dot(d['W_arr_inv'][i], \
                                            d['v_arr'][i] - \
                                            np.dot(d['B_arr'][i].T, d['theta_hat']))
            self.levels_dict[self.level_to_be_updated] = deepcopy(d)
            self.level_to_be_updated = -1
            self.update_level_flag = False
        #print(self.levels_dict)
        super().update(reward, regret, arm_set)
        self.t += 1