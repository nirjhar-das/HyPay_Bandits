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
        self.levels_dict = [{'V_inv': (1/self.lmbda)* np.eye(self.d + self.k*self.L),
                         'theta_hat': np.zeros((self.d + self.k*self.L,)),
                         'u': np.zeros((self.d + self.k*self.L,))} for _ in range(self.num_levels)]
        self.t = 0
        self.a_t = 0
        self.level_to_be_updated = -1
        self.update_level_flag = False
        self.modify_arms()
    
    def modify_arms(self):
        ls = []
        for i in range(self.L):
            a = np.zeros((self.d + self.k*self.L,))
            a[:self.d] = self.arms[i][0]
            a[self.d + self.k*i: self.d + self.k*(i+1)] = self.arms[i][1]
            ls.append(a)
        self.arms = ls
    
    def get_estimates_and_confidence(self, level, arm_idx):
        d = self.levels_dict[level]
        arm_estimates = {i: [] for i in arm_idx}
        for idx in arm_idx:
            mean = np.dot(self.arms[idx], d['theta_hat'])
            bonus = (np.sqrt(self.lmbda)*self.S + \
                     np.sqrt(2*np.log(self.T*self.L/self.delta))) * \
                    np.sqrt(np.dot(self.arms[idx], np.dot(d['V_inv'], self.arms[idx])))
            arm_estimates[idx] += [mean + bonus, bonus]
        
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
            x = self.arms[self.a_t]
            V_inv = deepcopy(d['V_inv'])
            Vx = np.dot(V_inv, x)
            V_inv -= np.outer(Vx, Vx)/(1 + np.dot(Vx, x))
            d['V_inv'] = V_inv
            d['u'] += (reward * x)
            d['theta_hat'] = np.dot(d['V_inv'], d['u'])
            self.levels_dict[self.level_to_be_updated] = deepcopy(d)
            self.level_to_be_updated = -1
            self.update_level_flag = False
        #print(self.levels_dict)
        super().update(reward, regret, arm_set)
        self.modify_arms()
        self.t += 1