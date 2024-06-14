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
from concurrent.futures import ProcessPoolExecutor as Pool

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
            self.file_names = self.file_names[1:]
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
        yield "Data Exhausted!"


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
    from algorithms.linear import DisLinUCB_Offline, HyLinUCB_Offline, OFUL_Offline, LinUCB_Offline, HyRan_Offline #, SupLinUCB_Offline
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


def run_bandit_simulation(folder, trial, T, model, name='Yahoo', output='./Results'):
    if name == 'Yahoo':
        algo_dict = {'LinUCB': {'lambda': 1e+3},
                     'DisLinUCB': {'lambda': 1e+2},
                     'HyLinUCB': {'lambda': 1e+3},
                     'HyRan': {'p': 0.5}}
        d, k, L = 36, 6, 20
        algo_arr = prepare_algo_arr(algo_dict, None, d, k, L)
        env = SemiSyntheticEnv(folder, 'Yahoo', 20)
        all_regrets = {key : [] for key in algo_dict.keys()}
        t = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rng = np.random.default_rng(trial)
        data_exhaust = False
        print('Simulating Bandit Learning...')
        for data in tqdm(env.step(), total=T, position=trial + 1):
            if data == "Data Exhausted!":
                data_exhaust = True
                break
            all_preds = [a.predict(data) for a in algo_arr]
            data_tensor = convert_to_long_tensors(data)
            data_tensor = data_tensor.float()
            data_tensor = data_tensor.to(device)
            with torch.no_grad():
                all_rewards = model(data_tensor).detach().cpu().numpy().reshape(-1)
            noisy_rewards = all_rewards + rng.normal(0.0, 0.00001, size=all_rewards.shape)
            max_reward = np.max(all_rewards)
            algo_regret_arr = [max_reward - all_rewards[a] for a in all_preds]
            for i, alg in enumerate(algo_arr):
                alg.update(data, noisy_rewards[i], algo_regret_arr[i])
                all_regrets[alg.name].append(algo_regret_arr[i])
            t += 1
            if t == T:
                break
        
        if data_exhaust:
            T = t
            print(f'Data Exhausted! Simulation done upto {T} time steps!')
        else:
            print('Simulation done.')
        df = pd.DataFrame(all_regrets)
        df.to_csv(os.path.join(output, f'Yahoo-Semi-Synthetic-{T}-{trial+1}-Final.csv'), index=False)
        return all_regrets

def plot_regret(regret_dict, name='Yahoo', output='./Results', id=None):
    if name == 'Yahoo':
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
        T = len(regret_dict[list(regret_dict.keys())[0]])
        time_steps = np.arange(1, T+1)
        for k in regret_dict.keys():
            ax.plot(time_steps, np.cumsum(regret_dict[k]), label=k, color=get_color(k))
        
        ax.legend()
        ax.grid()
        ax.set_title('Yahoo! Front Page')
        ax.set_xlabel('Time')
        ax.set_ylabel('Regret')
        if id is not None:
            plt.savefig(os.path.join(output, f'Yahoo-Semi-Synthetic-{T}-{id+1}-Final.png'), dpi=200)
        else:
            plt.savefig(os.path.join(output, f'Yahoo-Semi-Synthetic-{T}-Final.png'), dpi=200)

def run_sim_wrapper(args):
    return run_bandit_simulation(*args)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the dataset [options: Yahoo]', default='Yahoo')
    parser.add_argument('-z', '--zipfolder', type=str, help='path to zipped data folder')
    parser.add_argument('-s', '--samples_per_epoch', type=int, help='max number of samples to train per epoch', default=100000)
    parser.add_argument('-e', '--num_epochs', type=int, help='number of epochs', default=5)
    parser.add_argument('-T', '--timesteps', type=int, help='number of time steps', default=100000)
    parser.add_argument('-m', '--model', type=str, help='model type [options: Linear]', default='Linear')
    parser.add_argument('-o', '--output', type=str, help='output folder path', default='./Results')
    parser.add_argument('-t', '--num_trials', type=int, default=5, help='number of trials')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if args.name == 'Yahoo':
        d, k, L = 36, 6, 20
        if os.path.exists(os.path.join(args.output, f'lin_reg_model_{d}_{k}_{L}.pt')):
            print('Model found. Loading...')
            models = []
            for i in range(args.num_trials):
                model = LinearRegression(d, k, L)
                model = torch.load(os.path.join(args.output, f'lin_reg_model_{d}_{k}_{L}.pt'))
                model.eval()
                models.append(model) 
            print('Model loaded')
        else:
            print('Model not found. Training model...')
            model = train_linear_regression(args.zipfolder, n_epochs=args.num_epochs, max_samples_per_epoch=args.samples_per_epoch)
            print('Saving model...')
            torch.save(model, os.path.join(args.output, f'lin_reg_model_{d}_{k}_{L}.pt'))
            print('Model saved')
            models = []
            for i in range(args.num_trials):
                model = LinearRegression(d, k, L)
                model = torch.load(os.path.join(args.output, f'lin_reg_model_{d}_{k}_{L}.pt'))
                models.append(model) 
    
    args_arr = [[args.zipfolder, i, args.timesteps, models[i], args.name, args.output] for i in range(args.num_trials)]
    with Pool() as p:
        regret_dict_arr = p.map(run_sim_wrapper, args_arr)
    total_dict = {}
    for i, regret_dict in enumerate(regret_dict_arr):
        plot_regret(regret_dict, args.name, args.output, i)
        for k in regret_dict.keys():
            if k not in total_dict.keys():
                total_dict[k] = np.array(regret_dict[k])
            else:
                total_dict[k] += np.array(regret_dict[k])
    
    for k in total_dict.keys():
        total_dict[k] /= args.num_trials
    
    plot_regret(total_dict, args.name, args.output)
