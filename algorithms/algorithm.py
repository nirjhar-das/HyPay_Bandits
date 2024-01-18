import pandas as pd

class Algorithm:
    def __init__(self, name, arms):
        self.name = name
        self.arms = arms
        self.L = len(arms)
        self.d = arms[0][0].shape[0]
        self.k = arms[0][1].shape[0]
        self.rewards = []
        self.regrets = []

    def next_action(self):
        return 0
    
    def update(self, reward, regret, arm_set):
        self.rewards.append(reward)
        self.regrets.append(regret)
        self.arms = arm_set
    
    def save_results(self):
        df = pd.DataFrame(data = {'reward': self.rewards, 'regret': self.regrets})
        df.to_csv(f'{self.name}_Result.csv', index=False)