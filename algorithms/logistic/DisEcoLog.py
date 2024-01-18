from ..algorithm import Algorithm
from ...utils import sigmoid, dsigmoid

import numpy as np
import scipy as sc
import cvxpy as cp

class DisEcoLog(Algorithm):
    def __init__(self, arms, delta, M, N, S1, S2, lmbda, info=None):
        super().__init__(f'DisEcoLog_{info}' if info is not None else 'DisEcoLog', arms)
        self.M = np.sqrt(M*M + N*N)
        self.S = np.sqrt(S1*S1 + S2*S2)
        self.lmbda = lmbda
        self.delta = delta
        self.theta_hat_arr = []
        self.W_arr = []
        self.W_inv_arr = []
        self.V_arr = []
        self.v_arr = []
        self.t_i_arr = []
        self.warm_up_arr = []
        for i in range(self.L):
            self.theta_hat_arr.append(np.zeros((self.d + self.k,)))
            self.W_arr.append(self.lmbda * np.eye(self.d + self.k))
            self.W_inv_arr.append((1.0 / self.lmbda) * np.eye(self.d + self.k))
            self.v_arr.append(np.zeros((self.d + self.k,)))
            self.t_i_arr.append(0)
            self.V_arr.append((self.conf_radius(i)**2) * np.eye(self.d + self.k))
            self.warm_up_arr.append([])
        self.t = 0
        self.a_t = 0
        self.modify_arms()
        self.ecolog_problem = self.create_ecolog_problem_def()
        self.theta_bar_problem = self.create_theta_bar_problem_def()

    
    def modify_arms(self):
        ls = []
        for i in range(self.L):
            ls.append(np.concatenate((self.arms[i][0], self.arms[i][1])))
        self.arms = ls
    
    def conf_radius(self, idx):
        lmbda_t = (self.d + self.k)*np.log((4.0 + (self.t_i_arr[idx]+1)/4.0)/self.delta)
        sqrt_gamma_t = (self.S + 1.5) * np.sqrt(lmbda_t)
        return sqrt_gamma_t
    
    def get_ucb_index(self, idx):
        mean_reward = np.dot(self.arms[idx], self.theta_hat_arr[idx])
        bonus = self.conf_radius(idx) * np.dot(self.arms[idx], np.dot(self.W_inv_arr[idx], self.arms[idx]))
        return mean_reward + bonus

    def get_next_action(self):
        max_idx = self.get_ucb_index(0)
        self.a_t = 0
        for i in range(1, self.L):
            idx = self.get_ucb_index(i)
            if idx > max_idx:
                self.a_t = i
                max_idx = idx
        return self.a_t

    def create_ecolog_problem_def(self):
        theta = cp.Variable(self.d + self.k, name='theta')
        arm = cp.Parameter(self.d + self.k, name='arm')
        theta_hat = cp.Parameter(self.d + self.k, name='theta_hat')
        V = cp.Parameter((self.d + self.k, self.d + self.k), name='V_matrix')
        W = cp.Parameter((self.d + self.k, self.d + self.k), name='W_matrix')
        r_t = cp.Parameter(name='reward')
        gamma = cp.Parameter(name='gamma')
        ellipse = cp.quad_form(theta - theta_hat, V) <= gamma
        obj = - r_t * cp.dot(arm, theta) + cp.logistic(cp.dot(arm, theta)) +\
                (1.0 / (2.0 + self.S)) * cp.quad_form(theta - theta_hat, W)
        prob = cp.Problem(cp.Minimize(obj), ellipse)
        return prob

    def create_theta_bar_problem_def(self):
        theta = cp.Variable(self.d + self.k, name='theta')
        arm = cp.Parameter(self.d + self.k, name='arm')
        theta_hat = cp.Parameter(self.d + self.k, name='theta_hat')
        V = cp.Parameter((self.d + self.k, self.d + self.k), name='V_matrix')
        W = cp.Parameter((self.d + self.k, self.d + self.k), name='W_matrix')
        gamma = cp.Parameter(name='gamma')
        ellipse = cp.quad_form(theta - theta_hat, V) <= gamma
        obj = -cp.dot(arm, theta) + 2*cp.logistic(cp.dot(arm, theta)) +\
                (1.0 / (2.0 + self.S)) * cp.quad_form(theta - theta_hat, W)
        prob = cp.Problem(cp.Minimize(obj), ellipse)
        return prob
    
    def solve_warmup_problem(self, idx):
        theta = cp.Variable(self.d + self.k, name='theta')
        arm_matrix = np.array([a[0] for a in self.warm_up_arr[idx]])
        reward_matrix = np.array([a[1] for a in self.warm_up_arr[idx]])
        gamma = self.conf_radius(idx)**2
        obj = -cp.sum(cp.multiply(reward_matrix, arm_matrix @ theta) \
                       + cp.logistic(arm_matrix @ theta)) + gamma * cp.pnorm(theta, 2)**2
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()
        return theta.value
    
    def solve_ecolog_problem(self, idx, reward):
        self.ecolog_problem.param_dict['theta_hat'].value = self.theta_hat_arr[idx]
        self.ecolog_problem.param_dict['arm'].value = self.arms[idx]
        self.ecolog_problem.param_dict['W_matrix'].value = self.W_arr[idx]
        self.ecolog_problem.param_dict['V_matrix'].value = self.V_arr[idx]
        self.ecolog_problem.param_dict['reward'].value = reward
        self.ecolog_problem.param_dict['gamma'].value = self.conf_radius(idx)**2
        self.ecolog_problem.solve(solver=cp.SCS, eps=(1.0 / self.t_i_arr[idx]))
        return self.ecolog_problem.var_dict['theta'].value
    
    def solve_theta_bar_problem(self, idx):
        self.ecolog_problem.param_dict['theta_hat'].value = self.theta_hat_arr[idx]
        self.ecolog_problem.param_dict['arm'].value = self.arms[idx]
        self.ecolog_problem.param_dict['W_matrix'].value = self.W_arr[idx]
        self.ecolog_problem.param_dict['V_matrix'].value = self.V_arr[idx]
        self.ecolog_problem.param_dict['gamma'].value = self.conf_radius(idx)**2
        self.ecolog_problem.solve(solver=cp.SCS, eps=(1.0 / self.t_i_arr[idx]))
        return self.ecolog_problem.var_dict['theta'].value
    
    def update(self, reward, regret, arm_set):
        arm = self.arms[self.a_t]
        theta_0 = self.solve_ecolog_problem(self.a_t, 0.0)
        theta_1 = self.solve_ecolog_problem(self.a_t, 1.0)
        theta_bar = self.solve_theta_bar_problem(self.a_t)
        warm_up_flag = False
        if (dsigmoid(np.dot(arm, theta_0)) <= 2.0 * dsigmoid(np.dot(arm, theta_bar))) \
            and (dsigmoid(np.dot(arm, theta_1)) <= 2.0 * dsigmoid(np.dot(arm, theta_bar))):
            if reward == 1.0:
                self.theta_hat_arr[self.a_t] = theta_1
                self.W_arr[self.a_t] += dsigmoid(np.dot(theta_1, arm)) * np.outer(arm, arm)
                self.W_inv_arr[self.a_t] -= dsigmoid(np.dot(theta_1, arm)) * np.outer(np.dot(self.W_inv_arr[self.a_t], arm), \
                                                     np.dot(self.W_inv_arr[self.a_t], arm)) / (1 + dsigmoid(np.dot(theta_1, arm)) *\
                                                                                               np.dot(arm, np.dot(self.W_inv_arr[self.a_t], arm)))
            else:
                self.theta_hat_arr[self.a_t] = theta_0
                self.W_arr[self.a_t] += dsigmoid(np.dot(theta_0, arm)) * np.outer(arm, arm)
                self.W_inv_arr[self.a_t] -= dsigmoid(np.dot(theta_0, arm)) * np.outer(np.dot(self.W_inv_arr[self.a_t], arm), \
                                                     np.dot(self.W_inv_arr[self.a_t], arm)) / (1 + dsigmoid(np.dot(theta_1, arm)) *\
                                                                                               np.dot(arm, np.dot(self.W_inv_arr[self.a_t], arm)))

        else:
            warm_up_flag = True
            self.warm_up_arr[self.a_t].append((arm, reward))
            self.theta_hat_arr[self.a_t] = self.solve_warmup_problem(self.a_t)
            self.V_arr[self.a_t] += ((1.0 / self.kappa) * np.outer(arm, arm) - (self.conf_radius(self.a_t)**2) * np.eye(self.d + self.k))

        self.t_i_arr[self.a_t] += 1
        if warm_up_flag:
            self.V_arr[self.a_t] += (self.conf_radius(self.a_t)**2) * np.eye(self.d + self.k)
        
        super().update(reward, regret, arm_set)
        self.modify_arms()
        self.t += 1