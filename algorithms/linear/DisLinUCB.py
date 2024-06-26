from ..algorithm import Algorithm
import numpy as np
import pandas as pd

# sinhagaur88@gmail.com

class DisLinUCB(Algorithm):
    def __init__(self, arms, delta, M, N, S1, S2, sigma, lmbda, info=None):
        super().__init__(f'DisLinUCB_{info}' if info is not None else 'DisLinUCB', arms)
        self.M = np.sqrt(M*M + N*N)
        self.S = np.sqrt(S1*S1 + S2*S2)
        self.lmbda = lmbda
        self.delta = delta
        self.sigma = sigma
        self.theta_hat_arr = []
        self.W_arr = []
        self.W_arr_inv = []
        self.v_arr = []
        self.t_i_arr = []
        for i in range(self.L):
            self.theta_hat_arr.append(np.zeros((self.d + self.k,)))
            self.W_arr.append(self.lmbda * np.eye(self.d + self.k))
            self.W_arr_inv.append((1.0 / self.lmbda) * np.eye(self.d + self.k))
            self.v_arr.append(np.zeros((self.d + self.k,)))
            self.t_i_arr.append(0)
        self.t = 0
        self.a_t = 0
        self.modify_arms()
    
    def modify_arms(self):
        ls = []
        for i in range(self.L):
            ls.append(np.concatenate((self.arms[i][0], self.arms[i][1])))
        self.arms = ls

    def p_beta(self, i):
        p = self.S * np.sqrt(self.lmbda) +\
            self.sigma*np.sqrt(2*np.log(1/self.delta) + \
                    (self.d + self.k) * np.log(1 + (self.t_i_arr[i]*self.M*self.M)/(self.lmbda * (self.d + self.k))))
        return p
    
    def get_reward_estimate(self, i, a=None):
        if a is None:
            a = self.arms[i]
        reward = np.dot(a, self.theta_hat_arr[i]) +\
                    self.p_beta(i) * np.sqrt(np.dot(a, np.dot(self.W_arr_inv[i], a)))
        return reward

    def next_action(self):
        max_reward = self.get_reward_estimate(0)
        self.a_t = 0
        for i in range(1, self.L):
            reward = self.get_reward_estimate(i)
            if reward > max_reward:
                self.a_t = i
                max_reward = reward
        return self.a_t
    
    def update(self, reward, regret, arm_set):
        x_t_vec = self.arms[self.a_t]
        self.W_arr[self.a_t] += np.outer(x_t_vec, x_t_vec)
        x_W_inv = np.dot(self.W_arr_inv[self.a_t], x_t_vec)
        self.W_arr_inv[self.a_t] -= np.outer(x_W_inv, x_W_inv) / (1 + np.dot(x_t_vec, x_W_inv))
        self.v_arr[self.a_t] += reward * self.arms[self.a_t]
        self.theta_hat_arr[self.a_t] = np.dot(self.W_arr_inv[self.a_t], self.v_arr[self.a_t])
        super().update(reward, regret, arm_set)
        self.modify_arms()
        self.t += 1
        self.t_i_arr[self.a_t] += 1