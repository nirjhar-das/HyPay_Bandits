import numpy as np

class HyRan:
    def __init__(self, arms, p, lam=1):
        ##Initialization
        self.name = 'HyRan'
        self.L = len(arms)
        self.d_temp = arms[0][0].shape[0]
        self.k = arms[0][1].shape[0]
        self.d = self.d_temp + self.L*self.k
        self.t=0
        self.yx=np.zeros(self.d)
        self.f=np.zeros(self.d)
        self.Vinv=lam*np.eye(self.d)
        self.Binv=lam*np.eye(self.d)
        self.V = np.zeros((self.d,self.d))
        self.W = np.zeros((self.d,self.d))
        self.beta_hat=np.zeros(self.d)
        self.beta_tilde=np.zeros(self.d)
        self.ridgeVinv = np.eye(self.d)
        self.ridgef = np.zeros(self.d)
        self.psi_size = 1
        self.arms = arms
        self.modify_arms()
        self.rewards = []
        self.regrets = [] 

        ## Hyperparameters
        self.p = p
        self.lam = lam
        self.settings = {'lam':lam, 'p':p}
    
    def modify_arms(self):
        ls = []
        for i in range(self.L):
            a = np.zeros((self.d_temp + self.k*self.L,))
            a[:self.d_temp] = self.arms[i][0]
            a[self.d_temp + self.k*i: self.d_temp + self.k*(i+1)] = self.arms[i][1]
            ls.append(a)
        self.arms = ls

    def next_action(self):
        # contexts: list [X(1),...X(N)]
        contexts = np.copy(self.arms)
        self.t = self.t + 1
        self.lam = 2*self.d*np.log(self.t+1)
        N = len(contexts)
        means = np.array([np.dot(X, self.beta_hat) for X in contexts])
        a_t = np.argmax(means)
        pi = (1-self.p)/(N-1)*np.ones(N)
        pi[a_t] = self.p
        tilde_a_t = np.argmax(np.random.multinomial(1, pi, size=1))
        self.DR = (tilde_a_t == a_t)
        
       ## Update matrices
        if self.DR: #when tilde_a_t == a_t
            X = np.array(contexts)
            self.V = self.V + X.T @ X
            self.psi_size = self.psi_size + 1
            try:
                self.Vinv = np.linalg.inv(self.V + self.lam*np.eye(self.d))
                self.Binv = np.linalg.inv(self.V + N*self.lam*np.sqrt(self.psi_size)*self.lam*np.eye(self.d))
            except:
                for i in range(N):
                    self.Vinv = self.sherman_morrison(contexts[i], self.Vinv)
                    self.Binv = self.sherman_morrison(contexts[i], self.Binv)
            self.W = self.W + np.outer(contexts[a_t], contexts[a_t]) / self.p
            
        else: #when tilde_a_t != a_t
            self.Vinv = self.sherman_morrison(contexts[a_t], self.Vinv)
        self.X_a = contexts[a_t]
        
        return a_t

    def update(self, reward, regret, arms):
        # Update beta_tilde
        self.ridgeVinv = self.sherman_morrison(self.X_a, self.ridgeVinv)
        self.ridgef = self.ridgef + reward * self.X_a
        ridge = self.ridgeVinv @ self.ridgef
        if np.linalg.norm(ridge) > 1:
            ridge = ridge/np.linalg.norm(ridge)

        # Update beta_tilde and beta_hat
        if self.DR: #when tilde_a_t == a_t
            self.f = self.f + reward*self.X_a/self.p
            self.beta_tilde = ridge + self.Vinv @ (-self.lam*ridge - self.W @ ridge + self.f)
            self.beta_hat = self.beta_tilde + self.Vinv @ (-self.lam*self.beta_tilde - self.W @ self.beta_tilde + self.f)
        else: #when tilde_a_t != a_t
            self.f = self.f + reward*self.X_a
            self.beta_tilde = self.Binv @ self.f
            self.beta_hat = self.Vinv @ self.f
        
        self.arms = arms
        self.modify_arms()
        self.rewards.append(reward)
        self.regrets.append(regret)
    
    def sherman_morrison(self, X, V, w=1):
        X_dummy = np.dot(V, X)
        result = V - np.outer(X_dummy, X_dummy) / (1.0 + np.dot(X, X_dummy))
        #result = V-(w*np.einsum('ij,j,k,kl -> il', V, X, X, V))/(1.+w*np.einsum('i,ij,j ->', X, V, X))
        return result