from ..algorithm import Algorithm
import numpy as np
import scipy as sc
import pandas as pd


class OFUL(Algorithm):
    def __init__(self, arms, delta, M, N, S1, S2, sigma, lmbda, info=None):
        super().__init__(f'OFUL_{info}' if info is not None else 'OFUL', arms)
        self.M = np.sqrt(M*M + N*N)
        self.S = np.sqrt(S1*S1 + self.L*S2*S2)
        self.lmbda = lmbda
        self.delta = delta
        self.sigma = sigma
        self.theta_hat = np.zeros((self.d + self.k*self.L,))
        self.u = np.zeros((self.d + self.k*self.L,))
        self.V_inv = (1/self.lmbda)* np.eye(self.d + self.k*self.L)
        self.V = self.lmbda* np.eye(self.d + self.k*self.L)
        self.V_inv_norm = [[] for _ in range(self.L)]
        self.block_matrix_norm = [[] for _ in range(self.L)]
        self.I_plus_A = [[] for _ in range(self.L)]
        self.I_minus_A_by_2 = [[] for _ in range(self.L)]
        self.min_eig_val_A = []
        self.max_eig_val_A = []
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
                    (self.d + self.L * self.k) * np.log(1 + (self.t*self.M*self.M)/(self.lmbda * (self.d + self.L * self.k))))
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
    
    def update_norm_arr(self):
        tt = [self.V[0:self.d, 0:self.d]]
        for i in range(self.L):
            a = self.arms[i]
            self.V_inv_norm[i].append(np.dot(a, np.dot(self.V_inv, a)))
            p1 = np.dot(a[0:self.d], np.dot(self.V_inv[0:self.d, 0:self.d], a[0:self.d]))
            ith_idx_start, ith_idx_end = self.d + (i)*self.k, self.d + (i+1)*self.k
            tt.append(self.V[ith_idx_start:ith_idx_end, ith_idx_start:ith_idx_end])
            p2 = np.dot(a[ith_idx_start:ith_idx_end], np.dot(self.V_inv[ith_idx_start:ith_idx_end, ith_idx_start:ith_idx_end], a[ith_idx_start:ith_idx_end]))
            self.block_matrix_norm[i].append(p1 + p2)
        U = sc.linalg.block_diag(*tt)
        sqrt_U = sc.linalg.sqrtm(U)
        U_inv_sqrt = np.linalg.inv(sqrt_U)
        I_plus_A = U_inv_sqrt @ self.V @ U_inv_sqrt
        A = I_plus_A - np.eye(I_plus_A.shape[0])
        self.min_eig_val_A.append(np.linalg.eigvalsh(A)[0])
        self.max_eig_val_A.append(np.linalg.eigvalsh(A)[-1])
        I_minus_A_by_2 = np.eye(I_plus_A.shape[0]) - 0.5*A
        for i in range(self.L):
            a = self.arms[i]
            self.I_plus_A[i].append(np.dot(a, np.dot(I_plus_A,  a)))
            self.I_minus_A_by_2[i].append(np.dot(a, np.dot(U_inv_sqrt @ I_minus_A_by_2 @ U_inv_sqrt, a)))
    
    def update(self, reward, regret, arm_set):
        #self.update_norm_arr()
        Vx  = np.dot(self.V_inv, self.arms[self.a_t])
        self.V_inv -= np.outer(Vx, Vx)/(1 + np.dot(Vx, self.arms[self.a_t]))
        self.V += np.outer(self.arms[self.a_t], self.arms[self.a_t])
        self.u += reward * self.arms[self.a_t]
        self.theta_hat = np.dot(self.V_inv, self.u)
        super().update(reward, regret, arm_set)
        self.modify_arms()
        self.t += 1