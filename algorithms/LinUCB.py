from .algorithm import Algorithm
import numpy as np
import pandas as pd


class LinUCB(Algorithm):
    def __init__(self, arms, delta, M, N, S1, S2, sigma, lmbda, info=None):
        super().__init__(f'LinUCB_{info}' if info is not None else 'LinUCB', arms)
        self.M = np.sqrt(M*M + N*N)
        self.S = np.sqrt(S1*S1 + self.k*S2*S2)
        self.lmbda = lmbda
        self.delta = delta
        self.sigma = sigma
        self.theta_hat = np.zeros((self.d + self.k*self.L,))
        self.u = np.zeros((self.d + self.k*self.L,))
        self.V_inv = (1/self.lmbda)* np.eye(self.d + self.k*self.L)
        self.t = 0
        self.a_t = 0
        self.modify_arms()
    
    def modify_arms(self):
        ls = []
        for i in range(self.L):
            a = np.zeros((self.d + self.k*self.L,))
            a[:self.d] = self.arms[i][0]
            a[self.d + self.k*i: self.d + self.k*(i+1)] = self.arms[i][1]
            ls.append(a)
        self.arms = ls

    def conf_radius(self):
        p = self.S * np.sqrt(self.lmbda) +\
            self.sigma*np.sqrt(2*np.log(1/self.delta) + \
                    (self.d + self.k*self.L) * np.log(1 + (self.t*self.M*self.M)/(self.lmbda * (self.d + self.k*self.L))))
        return p
    
    def get_reward_estimate(self, i, a=None):
        if a is None:
            a = self.arms[i]
        reward = np.dot(a, self.theta_hat) +\
                    self.conf_radius() * np.sqrt(np.dot(a, np.dot(self.V_inv, a)))
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
        Vx  = np.dot(self.V_inv, self.arms[self.a_t])
        self.V_inv -= np.outer(Vx, Vx)/(1 + np.dot(Vx, self.arms[self.a_t]))
        self.u += reward * self.arms[self.a_t]
        self.theta_hat = np.dot(self.V_inv, self.u)
        super().update(reward, regret, arm_set)
        self.modify_arms()
        self.t += 1

    def save_results(self):
        df = pd.DataFrame(data = {'reward': self.rewards, 'regret': self.regrets})
        df.to_csv(f'{self.name}_Result.csv', index=False)