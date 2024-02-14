import numpy as np
from ..algorithm import Algorithm

class HyRan(Algorithm):
    def __init__(self, arms, p, info=None):
        super().__init__(f'HyRan_{info}' if info is not None else 'HyRan', arms)
        ##Initialization
        self.t = 1
        self.theta_hat = np.zeros_like(self.arms[0][0])
        self.beta_hat_arr = []
        self.theta_tilde = np.zeros_like(self.arms[0][0])
        self.beta_tilde_arr = []
        self.W_arr = []
        self.B_arr = []
        self.v_arr = []
        for _ in range(self.L):
            self.beta_hat_arr.append(np.zeros_like(self.arms[0][1]))
            self.beta_tilde_arr.append(np.zeros_like(self.arms[0][1]))
            self.W_arr.append(np.zeros((self.k, self.k)))
            self.B_arr.append(np.zeros((self.d, self.k)))
            self.v_arr.append(np.zeros_like(self.arms[0][1]))
        self.u = np.zeros_like(self.arms[0][0])
        self.V_tilde = np.zeros((self.d, self.d))
        self.psi_size = 1
        self.a_t = -1
        ## Hyperparameters
        self.lmbda = 1.0
        self.p = p
        self.settings = {'p':p}
        self.pi = (1-self.p)/(self.L-1)*np.ones(self.L)
    

    def next_action(self):
        means = np.array([np.dot(X1, self.theta_hat) + np.dot(X2, beta) for (X1, X2), beta, in zip(self.arms, self.beta_hat_arr)])
        self.a_t = np.argmax(means)

        return self.a_t
    
    def update(self, reward, regret, next_arms):
        pi = np.copy(self.pi)
        pi[self.a_t] = self.p
        tilde_a_t = np.argmax(np.random.multinomial(1, pi, size=1))
        self.DR = (tilde_a_t == self.a_t)
        
       ## Update matrices
        if self.DR: #when tilde_a_t == a_t
            for i in range(self.L):
                y_hat = (1 - ((i == tilde_a_t)/pi[i])) * (np.dot(self.arms[i][0], self.theta_tilde) + np.dot(self.arms[i][1], self.beta_tilde_arr[i])) \
                            + ((i == tilde_a_t)/pi[i]) * reward
                self.matrix_update(i, y_hat)

            self.psi_size = self.psi_size + 1
            
        else: #when tilde_a_t != a_t
            self.matrix_update(self.a_t, reward)
        
        self.parameter_update()
        self.t += 1
        self.lmbda = 2*(self.d + self.k*self.L)*np.log(self.t+1)
        super().update(reward, regret, next_arms)


    def matrix_update(self, idx, r):
        self.u += r * self.arms[idx][0]
        self.v_arr[idx] += r * self.arms[idx][1]
        self.V_tilde += np.outer(self.arms[idx][0], self.arms[idx][0])
        self.W_arr[idx] += np.outer(self.arms[idx][1], self.arms[idx][1])

    
    def parameter_update(self):
        dummy_u_1 = self.u
        dummy_u_2 = self.u
        VV_1 = self.V_tilde
        VV_2 = self.V_tilde
        for i in range(self.L):
            dummy_u_1 -= np.dot(self.B_arr[i] @ np.linalg.inv(self.W_arr[i] + self.lmbda * np.eye(self.k)), self.v_arr[i]).reshape(-1)
            dummy_u_2 -= np.dot(self.B_arr[i] @ np.linalg.inv(self.W_arr[i] + np.sqrt(self.t) * np.eye(self.k)), self.v_arr[i]).reshape(-1)
            VV_1 -= self.B_arr[i] @ np.linalg.inv(self.W_arr[i] + self.lmbda * np.eye(self.k)) @ self.B_arr[i].T
            VV_2 -= self.B_arr[i] @ np.linalg.inv(self.W_arr[i] + np.sqrt(self.t) * np.eye(self.k)) @ self.B_arr[i].T
        self.theta_hat = np.dot(np.linalg.inv(VV_1 + self.lmbda * np.eye(self.d)), dummy_u_1)
        self.theta_tilde = np.dot(np.linalg.inv(VV_2 + np.sqrt(self.t) * np.eye(self.d)), dummy_u_2)
        for i in range(self.L):
            self.beta_hat_arr[i] = np.dot(np.linalg.inv(self.W_arr[i] + self.lmbda * np.eye(self.k)),
                                        self.v_arr[i] - \
                                        np.dot(self.B_arr[i].T, self.theta_hat))
            self.beta_tilde_arr[i] = np.dot(np.linalg.inv(self.W_arr[i] + np.sqrt(self.t) * np.eye(self.k)),
                                        self.v_arr[i] - \
                                        np.dot(self.B_arr[i].T, self.theta_tilde))
        

        