from ..algorithm import Algorithm
from .utils import sigmoid, dsigmoid

import numpy as np
import scipy as sc
import cvxpy as cp

class MHyEcoLog(Algorithm):
    def __init__(self, arms, delta, M, N, S1, S2, lmbda, kappa, info=None):
        super().__init__(f'MHyEcoLog_{info}' if info is not None else 'MHyEcoLog', arms)
        self.M = np.sqrt(M*M + N*N)
        self.S = np.sqrt(S1*S1 + S2*S2)
        self.lmbda = lmbda
        self.kappa = kappa
        self.delta = delta
        self.t = 0
        self.theta_hat = np.zeros(self.d + self.L * self.k)
        self.W = self.lmbda * np.eye(self.d + self.L * self.k)
        self.W_inv = (1.0 / self.lmbda) * np.eye(self.d + self.L * self.k)
        self.V = (self.conf_radius()**2) * np.eye(self.d + self.L * self.k)
        self.warm_up_arr = []
        self.a_t = 0
        self.modify_arms()
        self.ecolog_problem = self.create_ecolog_problem_def()
        self.theta_bar_problem = self.create_theta_bar_problem_def()

    def modify_arms(self):
        ls = []
        for i in range(self.L):
            a = np.zeros((self.d + self.k*self.L,))
            a[:self.d] = self.arms[i][0]
            a[self.d + self.k*i: self.d + self.k*(i+1)] = self.arms[i][1]
            ls.append(a)
        self.arms = ls
    
    def conf_radius(self):
        lmbda_t = (self.d + self.k)*np.log((4.0 + (self.t+1)/4.0)/self.delta)
        sqrt_gamma_t = (self.S + 1.5) * np.sqrt(lmbda_t)
        return sqrt_gamma_t
    
    def get_ucb_index(self, idx):
        mean_reward = np.dot(self.arms[idx], self.theta_hat)
        bonus = self.conf_radius() * np.sqrt(np.dot(self.arms[idx], np.dot(self.W_inv, self.arms[idx])))
        return mean_reward + bonus

    def next_action(self):
        max_idx = self.get_ucb_index(0)
        self.a_t = 0
        for i in range(1, self.L):
            idx = self.get_ucb_index(i)
            if idx > max_idx:
                self.a_t = i
                max_idx = idx
        return self.a_t

    def create_ecolog_problem_def(self):
        theta = cp.Variable((self.d + self.L * self.k, 1), name='theta')
        theta_proxy = cp.Variable((self.d + self.L * self.k, 1), name='theta_proxy')
        arm = cp.Parameter((self.d + self.L * self.k, 1), name='arm')
        theta_hat = cp.Parameter((self.d + self.L * self.k, 1), name='theta_hat')
        V_sqrt = cp.Parameter((self.d + self.L * self.k, self.d + self.L * self.k), name='V_matrix', symmetric=True)
        W_sqrt = cp.Parameter((self.d + self.L * self.k, self.d + self.L * self.k), name='W_matrix', symmetric=True)
        r_t = cp.Parameter(name='reward')
        gamma = cp.Parameter(name='gamma')
        ellipse = cp.sum_squares(V_sqrt @ theta_proxy) <= gamma
        equality_1 = theta_proxy == theta - theta_hat
        theta_proxy_rt = cp.Variable((self.d + self.L * self.k, 1), name='theta_proxy_rt')
        equality_2 = theta_proxy_rt == r_t * theta
        obj = - cp.sum(arm.T @ theta_proxy_rt) + cp.logistic(cp.sum(arm.T @ theta)) +\
                (1.0 / (2.0 + self.S)) * cp.sum_squares(W_sqrt @ theta_proxy)
        prob = cp.Problem(cp.Minimize(obj), [ellipse, equality_1, equality_2])
        return prob

    def create_theta_bar_problem_def(self):
        theta = cp.Variable((self.d + self.L * self.k, 1), name='theta')
        theta_proxy = cp.Variable((self.d + self.L * self.k, 1), name='theta_proxy')
        arm = cp.Parameter((self.d + self.L * self.k, 1), name='arm')
        theta_hat = cp.Parameter((self.d + self.L * self.k, 1), name='theta_hat')
        V_sqrt = cp.Parameter((self.d + self.L * self.k, self.d + self.L * self.k), name='V_matrix', symmetric=True)
        W_sqrt = cp.Parameter((self.d + self.L * self.k, self.d + self.L * self.k), name='W_matrix', symmetric=True)
        gamma = cp.Parameter(name='gamma')
        ellipse = cp.sum_squares(V_sqrt @ theta_proxy) <= gamma
        equality = theta_proxy == theta - theta_hat
        obj = -cp.sum(arm.T @ theta) + 2*cp.logistic(cp.sum(arm.T @ theta)) +\
                (1.0 / (2.0 + self.S)) * cp.sum_squares(W_sqrt @ theta_proxy)
        prob = cp.Problem(cp.Minimize(obj), [ellipse, equality])
        return prob
    
    def solve_warmup_problem(self):
        theta = cp.Variable((self.d + self.L * self.k), name='theta')
        arm_matrix = np.array([a[0] for a in self.warm_up_arr])
        reward_matrix = np.array([a[1] for a in self.warm_up_arr])
        gamma = self.conf_radius()**2
        obj = - cp.sum(cp.multiply(reward_matrix, arm_matrix @ theta) \
                       - cp.logistic(arm_matrix @ theta)) + gamma * cp.sum_squares(theta)
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve(solver=cp.SCS, eps=1e-5)
        return theta.value
    
    def solve_ecolog_problem(self, idx, reward):
        self.ecolog_problem.param_dict['theta_hat'].value = self.theta_hat.reshape(-1 , 1)
        self.ecolog_problem.param_dict['arm'].value = self.arms[idx].reshape(-1, 1)
        self.ecolog_problem.param_dict['W_matrix'].value = sc.linalg.sqrtm(self.W)
        self.ecolog_problem.param_dict['V_matrix'].value = sc.linalg.sqrtm(self.V)
        self.ecolog_problem.param_dict['reward'].value = reward
        self.ecolog_problem.param_dict['gamma'].value = self.conf_radius()**2
        self.ecolog_problem.solve(solver=cp.SCS, eps=(1.0 / (self.t + 1.0)))
        return self.ecolog_problem.var_dict['theta'].value
    
    def solve_theta_bar_problem(self, idx):
        self.ecolog_problem.param_dict['theta_hat'].value = self.theta_hat.reshape(-1, 1)
        self.ecolog_problem.param_dict['arm'].value = self.arms[idx].reshape(-1, 1)
        self.ecolog_problem.param_dict['W_matrix'].value = sc.linalg.sqrtm(self.W)
        self.ecolog_problem.param_dict['V_matrix'].value = sc.linalg.sqrtm(self.V)
        self.ecolog_problem.param_dict['gamma'].value = self.conf_radius()**2
        self.ecolog_problem.solve(solver=cp.SCS, eps=(1.0 / (self.t + 1.0)))
        return self.ecolog_problem.var_dict['theta'].value
    
    def update(self, reward, regret, arm_set):
        arm = self.arms[self.a_t]
        theta_0 = self.solve_ecolog_problem(self.a_t, 0.0).reshape(-1)
        theta_1 = self.solve_ecolog_problem(self.a_t, 1.0).reshape(-1)
        theta_bar = self.solve_theta_bar_problem(self.a_t).reshape(-1)
        warm_up_flag = False
        if (dsigmoid(np.dot(arm, theta_0)) <= 2.0 * dsigmoid(np.dot(arm, theta_bar))) \
            and (dsigmoid(np.dot(arm, theta_1)) <= 2.0 * dsigmoid(np.dot(arm, theta_bar))):
            if reward == 1.0:
                self.theta_hat = theta_1
                self.W += dsigmoid(np.dot(theta_1, arm)) * np.outer(arm, arm)
                self.W_inv -= dsigmoid(np.dot(theta_1, arm)) * np.outer(np.dot(self.W_inv, arm), \
                                                     np.dot(self.W_inv, arm)) / (1 + dsigmoid(np.dot(theta_1, arm)) *\
                                                                                               np.dot(arm, np.dot(self.W_inv, arm)))
            else:
                self.theta_hat = theta_0
                self.W += dsigmoid(np.dot(theta_0, arm)) * np.outer(arm, arm)
                self.W_inv -= dsigmoid(np.dot(theta_0, arm)) * np.outer(np.dot(self.W_inv, arm), \
                                                     np.dot(self.W_inv, arm)) / (1 + dsigmoid(np.dot(theta_0, arm)) *\
                                                                                               np.dot(arm, np.dot(self.W_inv, arm)))

        else:
            warm_up_flag = True
            print(self.name, f': warm up required at round {self.t+1}')
            self.warm_up_arr.append((arm, reward))
            self.theta_hat = self.solve_warmup_problem()
            self.V += ((1.0 / self.kappa) * np.outer(arm, arm) - (self.conf_radius()**2) * np.eye(self.d + self.L * self.k))

        self.t += 1
        if warm_up_flag:
            self.V += (self.conf_radius()**2) * np.eye(self.d + self.L * self.k)
        
        super().update(reward, regret, arm_set)
        self.modify_arms()