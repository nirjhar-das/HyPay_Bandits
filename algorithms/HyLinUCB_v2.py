from .algorithm import Algorithm
import numpy as np
import pandas as pd

class HyLinUCBv2(Algorithm):
    def __init__(self, arms, delta, M, N, S1, S2, lmbda, gamma, info=None):
        super().__init__(f'HyLinUCBv2_{info}' if info is not None else 'HyLinUCBv2', arms)
        self.M = M
        self.N = N
        self.S1 = S1
        self.S2 = S2
        self.lmbda = lmbda
        self.gamma = gamma
        self.delta = delta
        self.theta_hat = np.zeros_like(self.arms[0][0])
        self.beta_hat_arr = []
        self.W_arr = []
        self.B_arr = []
        self.v_arr = []
        self.t_i_arr = []
        for i in range(self.L):
            self.beta_hat_arr.append(np.zeros_like(self.arms[0][1]))
            self.W_arr.append(self.gamma * np.eye(self.k))
            self.B_arr.append(np.zeros((self.d, self.k)))
            self.v_arr.append(np.zeros_like(self.arms[0][1]))
            self.t_i_arr.append(0)
        self.u = np.zeros_like(self.arms[0][0])
        self.V_tilde = self.lmbda * np.eye(self.d)
        self.t = 0
        self.a_t = 0
    
    def q_theta(self):
        q = self.S1 * np.sqrt(self.lmbda) +\
            np.sqrt(2*np.log(1/self.delta) + \
                    self.d * np.log(1 + (self.t*self.M*self.M)/(self.lmbda * self.d))) +\
                    np.sqrt(2 * self.d * self.L * self.S2)
        return q
    
    def get_reward_estimate(self, i):
        x_tilde = self.arms[i][0] - np.dot(self.B_arr[i] @ np.linalg.inv(self.W_arr[i]), self.arms[i][1])
        reward = np.dot(self.arms[i][0], self.theta_hat) +\
                    np.dot(self.arms[i][1], self.beta_hat_arr[i]) +\
                    self.q_theta() * np.sqrt(np.dot(x_tilde, np.dot(np.linalg.inv(self.V_tilde), x_tilde)))
        return reward

    def next_action(self):
        max_reward = self.get_reward_estimate(0)
        self.a_t = 0
        for i in range(1, self.L):
            reward = self.get_reward_estimate(i)
            if reward > max_reward:
                self.a_t = i
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
        self.theta_hat = np.dot(np.linalg.inv(self.V_tilde), \
                        self.u - np.dot(self.B_arr[self.a_t] @ np.linalg.inv(self.W_arr[self.a_t]), \
                            self.v_arr[self.a_t]))
        for i in range(self.L):
            self.beta_hat_arr[i] = np.dot(np.linalg.inv(self.W_arr[i]), \
                                        self.v_arr[i] - \
                                        np.dot(self.B_arr[i].T, self.theta_hat))
        super().update(reward, regret, arm_set)
        self.t += 1

    def save_results(self):
        df = pd.DataFrame(data = {'reward': self.rewards, 'regret': self.regrets})
        df.to_csv(f'{self.name}_Result.csv', index=False)