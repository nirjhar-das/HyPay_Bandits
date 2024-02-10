import os
import gzip
import argparse
from itertools import cycle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import get_color

class YahooDataset(IterableDataset):
    def __init__(self, folder):
        super(IterableDataset).__init__()
        self.num_arms = 20
        self.file_names = []
        for root, _, files in os.walk(folder):
            for name in files:
                if '2009' in name:
                    self.file_names.append(os.path.join(root, name))
        self.file_names = self.file_names[:5]
    
    def process_line(self, line):
        stream = line.split('|')
        arm_feat = np.zeros((6,))
        user_feat = np.zeros((6,))
        arm_idx = None
        a_t = None
        for i, data in enumerate(stream):
            if i == 0:
                ls = data.split(' ')
                a_t = ls[1]
                r_t = float(ls[2])
            if i == 1:
                ls = data.strip().split(' ')
                for token in ls:
                    if ':' in token:
                        key, val = token.split(':')
                        user_feat[int(key)-1] = float(val)
            elif i > 1:
                arm_feat = np.zeros((6,))
                ls = data.strip().split(' ')
                if ls[0] == a_t:
                    arm_idx = i - 2
                    for token in ls:
                        if ':' in token:
                            key, val = token.split(':')
                            arm_feat[int(key)-1] = float(val)
                    if arm_idx >= self.num_arms:
                        arm_idx = np.random.randint(self.num_arms)
                    tilde_x = np.zeros((36 + self.num_arms*6))
                    tilde_x[:36] = np.outer(user_feat, arm_feat).reshape(-1)
                    tilde_x[36 + arm_idx*6: 36 + (arm_idx + 1)*6] = arm_feat

        return tilde_x, r_t
        

    def __iter__(self):
        for name in cycle(self.file_names):
            with gzip.open(name, 'rt') as f:
                for line in f:
                    x, y = self.process_line(line)
                    x = torch.from_numpy(x)
                    y = torch.tensor(y)
                    yield x, y


class LinearRegression(nn.Module):
    def __init__(self, d, k, L):
        super(LinearRegression, self).__init__()
        self.params = nn.Linear(d + k*L, 1, bias=False)
        
    
    def forward(self, x):
        y_hat = self.params(x)
        return y_hat

class SemiSyntheticEnv:
    def __init__(self, dataset_folder, dataset_name, num_arms):
        if dataset_name == 'Yahoo':
            self.name = 'yahoo'
            self.file_names = []
            for root, _, files in os.walk(dataset_folder):
                for name in files:
                    if '2009' in name:
                        self.file_names.append(os.path.join(root, name))
            self.file_names[-1]
        self.num_arms = num_arms

    def process_line(self, line):
        if self.name == 'yahoo':
            stream = line.split('|')
            arm_list = []
            arm_id = []
            a_t = None
            for i, data in enumerate(stream):
                if i == 0:
                    ls = data.split(' ')
                    a_t = ls[1]
                    r_t = float(ls[2])
                if i == 1:
                    ls = data.strip().split(' ')
                    user_feat = np.zeros((6,))
                    for token in ls:
                        if ':' in token:
                            key, val = token.split(':')
                            user_feat[int(key)-1] = float(val)
                elif i > 1:
                    arm_feat = np.zeros((6,))
                    ls = data.strip().split(' ')
                    for token in ls:
                        if ':' in token:
                            key, val = token.split(':')
                            arm_feat[int(key)-1] = float(val)
                        else:
                            arm_id.append(token)
                    arm_list.append(arm_feat)
                if len(arm_list) == self.num_arms:
                    break
            
            if len(arm_list) < self.num_arms:
                for i in range(self.num_arms - len(arm_list)):
                    arm_list.append(np.zeros((6,)))
            
            hybrid_feat_list = []
            for i in range(self.num_arms):
                x_i = np.outer(user_feat, arm_list[i]).reshape(-1)
                z_i = arm_list[i]
                hybrid_feat_list.append((x_i, z_i))

            return hybrid_feat_list
    
        
    def step(self):
        for name in self.file_names:
            with gzip.open(name, 'rt') as f:
                for line in f:
                    yield self.process_line(line)


def convert_to_long_tensors(arm_features_list, name='yahoo'):
        if name == 'yahoo':
            d, k, L = 36, 6, 20
        X = np.zeros((L, d + k*L))
        for i in range(L):
            X[i][:d] = arm_features_list[i][0]
            X[i][d + i*k : d + (i+1)*k] = arm_features_list[i][1]
        return torch.from_numpy(X)


def train_linear_regression(folder, name='Yahoo', n_epochs=10, max_samples_per_epoch=100000, batch_size=4096):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if name == 'Yahoo':
        model = LinearRegression(36, 6, 20)
        model.to(device)
        dataset = YahooDataset(folder)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            print('Epoch:', epoch+1)
            running_loss = 0.0
            tot_size = 0
            for x, y in tqdm(dataloader, total=max_samples_per_epoch // batch_size):
                x = x.float()
                x = x.to(device)
                y = y.to(device)
                y = y.reshape(-1, 1)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                running_loss += loss.item()*x.size(dim=0)
                tot_size += x.size(dim=0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if tot_size > max_samples_per_epoch:
                    break
            print('Epoch loss:', running_loss / tot_size)
            
    return model

def prepare_algo_arr(algo_dict, T, d, k, L, delta=0.01):
    from algorithms.linear import DisLinUCB_Offline, HyLinUCB_Offline, OFUL_Offline, MHyLinUCB_Offline, LinUCB_Offline, HyRan_Offline #, SupLinUCB_Offline
    algo_arr = []
    for key in algo_dict.keys():
        if key == 'DisLinUCB':
            lmbda = algo_dict[key]['lambda']
            algo_arr.append(DisLinUCB_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.01, lmbda))
        elif key == 'HyLinUCB':
            lmbda = algo_dict[key]['lambda']
            algo_arr.append(HyLinUCB_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.01, lmbda))
        elif key == 'OFUL':
            lmbda = algo_dict[key]['lambda']
            algo_arr.append(OFUL_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.01, lmbda))
        elif key == 'MHyLinUCB':
            lmbda = algo_dict[key]['lambda']
            algo_arr.append(MHyLinUCB_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.01, lmbda))
        elif key == 'LinUCB':
            lmbda = algo_dict[key]['lambda']
            algo_arr.append(LinUCB_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.01, lmbda))
        elif key == 'HyRan':
            #lmbda = algo_dict[key]['lambda']
            p = algo_dict[key]['p']
            algo_arr.append(HyRan_Offline(d, k, L, p))
        # elif key == 'SupLinUCB':
        #     lmbda = algo_dict[key]['lambda']
        #     algo_arr.append(SupLinUCB_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.25, lmbda, T))
    return algo_arr


def run_bandit_simulation(folder, T, name='Yahoo', n_epochs=5, max_samples_per_epoch=100000, output='./Results'):
    if name == 'Yahoo':
        algo_dict = {'LinUCB': {'lambda': 0.001},
                     'DisLinUCB': {'lambda': 0.001},
                     'HyLinUCB': {'lambda': 0.0001},
                     'HyRan': {'p': 0.5}}
        d, k, L = 36, 6, 20
        algo_arr = prepare_algo_arr(algo_dict, None, d, k, L)
        if os.path.exists(os.path.join(output, f'lin_reg_model_{d}_{k}_{L}.pt')):
            print('Model found. Loading...')
            model = LinearRegression(d, k, L)
            model = torch.load(os.path.join(output, f'lin_reg_model_{d}_{k}_{L}.pt'))
            print('Model loaded')
        else:
            print('Model not found. Training model...')
            model = train_linear_regression(folder, n_epochs=n_epochs, max_samples_per_epoch=max_samples_per_epoch)
            print('Saving model...')
            torch.save(model, os.path.join(output, f'lin_reg_model_{d}_{k}_{L}.pt'))
            print('Model saved')
        model.eval()
        env = SemiSyntheticEnv(folder, 'Yahoo', 20)
        all_regrets = {key : [] for key in algo_dict.keys()}
        t = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rng = np.random.default_rng(0)
        print('Simulating Bandit Learning...')
        for data in tqdm(env.step(), total=T):
            all_preds = [a.predict(data) for a in algo_arr]
            data_tensor = convert_to_long_tensors(data)
            data_tensor = data_tensor.float()
            data_tensor = data_tensor.to(device)
            with torch.no_grad():
                all_rewards = model(data_tensor).detach().cpu().numpy().reshape(-1)
            noisy_rewards = all_rewards + rng.uniform(-0.1, 0.1, size=all_rewards.shape)
            max_reward = np.max(all_rewards)
            algo_regret_arr = [max_reward - all_rewards[a] for a in all_preds]
            for i, alg in enumerate(algo_arr):
                alg.update(data, noisy_rewards[i], algo_regret_arr[i])
                all_regrets[alg.name].append(algo_regret_arr[i])
            t += 1
            # if (t % 500) == 0:+
            #     print(data_tensor)
            if t == T:
                break
        print('Simulation done.')
        df = pd.DataFrame(all_regrets)
        df.to_csv(os.path.join(output, f'Yahoo-Semi-Synthetic-Regrets-{T}-Final.csv'), index=False)
        return all_regrets

def plot_regret(regret_dict, T, name='Yahoo', output='./Results'):
    if name == 'Yahoo':
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
        time_steps = np.arange(1, T+1)
        for k in regret_dict.keys():
            ax.plot(time_steps, np.cumsum(regret_dict[k]), label=k, color=get_color(k))
        
        ax.legend()
        ax.grid()
        ax.set_title('Yahoo! Front Page')
        ax.set_xlabel('Time')
        ax.set_ylabel('Regret')

        plt.savefig(os.path.join(output, f'Yahoo-Semi-Synthetic-{T}-Final.png'), dpi=200)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the dataset [options: Yahoo]', default='Yahoo')
    parser.add_argument('-z', '--zipfolder', type=str, help='path to zipped data folder')
    parser.add_argument('-s', '--samples_per_epoch', type=int, help='max number of samples to train per epoch', default=100000)
    parser.add_argument('-e', '--num_epochs', type=int, help='number of epochs', default=5)
    parser.add_argument('-T', '--timesteps', type=int, help='number of time steps', default=100000)
    parser.add_argument('-m', '--model', type=str, help='model type [options: Linear]', default='Linear')
    parser.add_argument('-o', '--output', type=str, help='output folder path', default='./Results')
    args = parser.parse_args()
                
    regret_dict = run_bandit_simulation(args.zipfolder, args.timesteps, n_epochs=args.num_epochs, name=args.name, max_samples_per_epoch=args.samples_per_epoch)
    plot_regret(regret_dict, args.timesteps, args.name, args.output)
    #model = LinearRegression(36, 6, 20)
    #model = torch.load(os.path.join(args.output, 'lin_reg_model.pt'))
    #print(model.params.weight)



#python semi_synthetic_exp.py -T 20000 -z ./Dataset/Yahoo-Front-Page/R6 -s 8192 -e 20