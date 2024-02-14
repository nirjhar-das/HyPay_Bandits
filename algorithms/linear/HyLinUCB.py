from ..algorithm import Algorithm
import numpy as np
import pandas as pd

class HyLinUCB(Algorithm):
    def __init__(self, arms, delta, M, N, S1, S2, sigma, lmbda, info=None):
        super().__init__(f'HyLinUCB_{info}' if info is not None else 'HyLinUCB', arms)
        self.M = M
        self.N = N
        self.M = np.sqrt(M*M + N*N)
        self.S = np.sqrt(S1*S1 + S2*S2)
        self.lmbda = lmbda
        self.delta = delta
        self.sigma = sigma
        self.theta_hat = np.zeros_like(self.arms[0][0])
        self.beta_hat_arr = []
        self.W_arr = []
        self.B_arr = []
        self.v_arr = []
        for i in range(self.L):
            self.beta_hat_arr.append(np.zeros_like(self.arms[0][1]))
            self.W_arr.append(self.lmbda * np.eye(self.k))
            self.B_arr.append(np.zeros((self.d, self.k)))
            self.v_arr.append(np.zeros_like(self.arms[0][1]))
        self.u = np.zeros_like(self.arms[0][0])
        self.V_tilde = self.lmbda * np.eye(self.d)
        self.t = 0
        self.a_t = 0

    def conf_radius(self):
        p = self.S * np.sqrt(self.lmbda) +\
            self.sigma*np.sqrt(2*np.log(1/self.delta) + \
                    (self.d + self.k) * np.log(1 + (self.t*self.M*self.M)/(self.lmbda * (self.d + self.k))))
        return p
    
    def ucb_bonus(self, i, a=None):
        if a is None:
            a = self.arms[i]
        V_inv = np.linalg.inv(self.V_tilde)
        W_inv = np.linalg.inv(self.W_arr[i])
        s_i = np.dot(a[0], np.dot(V_inv, a[0])) -\
                2*np.dot(a[0], np.dot(V_inv @ self.B_arr[i] @ W_inv, a[1])) +\
                np.dot(a[1], np.dot(W_inv, a[1])) +\
                np.dot(a[1], np.dot(W_inv @ self.B_arr[i].T @ V_inv @ self.B_arr[i] @ W_inv, a[1]))
        return self.conf_radius() * np.sqrt(s_i)
    
    
    def get_reward_estimate(self, i, a=None):
        if a is None:
            a = self.arms[i]
        reward = np.dot(a[0], self.theta_hat) +\
                    np.dot(a[1], self.beta_hat_arr[i]) +\
                    self.ucb_bonus(i, a)
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
        x_t_vec = self.arms[self.a_t][0].reshape(-1, 1)
        z_t_vec = self.arms[self.a_t][1].reshape(-1, 1)
        self.u += np.dot(self.B_arr[self.a_t] @ np.linalg.inv(self.W_arr[self.a_t]), \
                   self.v_arr[self.a_t])
        self.V_tilde = self.V_tilde + \
                        self.B_arr[self.a_t] @ np.linalg.inv(self.W_arr[self.a_t]) @ self.B_arr[self.a_t].T
        self.B_arr[self.a_t] += x_t_vec @ z_t_vec.T
        self.W_arr[self.a_t] += z_t_vec @ z_t_vec.T
        self.v_arr[self.a_t] += reward * z_t_vec.reshape(-1)
        self.V_tilde += (x_t_vec @ x_t_vec.T - \
                         self.B_arr[self.a_t] @ np.linalg.inv(self.W_arr[self.a_t]) @ self.B_arr[self.a_t].T)
        self.u += reward * x_t_vec.reshape(-1) - \
                    np.dot(self.B_arr[self.a_t] @ np.linalg.inv(self.W_arr[self.a_t]), self.v_arr[self.a_t]).reshape(-1)
        self.theta_hat = np.dot(np.linalg.inv(self.V_tilde), self.u)
        for i in range(self.L):
            self.beta_hat_arr[i] = np.dot(np.linalg.inv(self.W_arr[i]), \
                                        self.v_arr[i] - \
                                        np.dot(self.B_arr[i].T, self.theta_hat))
        super().update(reward, regret, arm_set)
        self.t += 1