import pandas as pd
import os

class Algorithm:
    def __init__(self, name, arms=None, d=None, k=None, L=None):
        self.name = name
        if arms is not None:
            self.arms = arms
            self.L = len(arms)
            self.d = arms[0][0].shape[0]
            self.k = arms[0][1].shape[0]
        else:
            self.d = d
            self.k = k
            self.L = L
        self.rewards = []
        self.regrets = []

    def next_action(self):
        return 0
    
    def update(self, reward, regret=None, arm_set=None):
        self.rewards.append(reward)
        if regret is not None:
            self.regrets.append(regret)
        if arm_set is not None:
            self.arms = arm_set
    
    def save_results(self, info, folder):
        if len(self.regrets) > 0:
            df = pd.DataFrame(data = {'reward': self.rewards, 'regret': self.regrets})
        else:
            df = pd.DataFrame(data = {'reward': self.rewards})
        df.to_csv(os.path.join(folder,  f'{info}_{self.name}_Result.csv'), index=False)