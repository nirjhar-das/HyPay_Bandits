import gzip
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import argparse
from multiprocessing.pool import Pool
import multiprocessing
import threading
import time


class OffEval:
    def __init__(self, dataset_folder, dataset_name, num_arms):
        if dataset_name == 'Yahoo':
            self.name = 'yahoo'
            self.file_names = []
            for root, _, files in os.walk(dataset_folder):
                for name in files:
                    if '2009' in name:
                        self.file_names.append(os.path.join(root, name))

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
            if len(arm_list) > self.num_arms:
                # if self.max_arms < len(arm_list):
                #     self.max_arms = len(arm_list)
                while True:
                    if self.num_arms == len(arm_list):
                        break
                    idx_to_del = np.random.randint(len(arm_list))
                    item_id = arm_id[idx_to_del]
                    if item_id == a_t:
                        continue
                    del arm_list[idx_to_del]
                    del arm_id[idx_to_del]
            elif len(arm_list) < self.num_arms:
                # if self.min_arms > len(arm_list):
                #     self.min_arms = len(arm_list)
                for i in range(self.num_arms - len(arm_list)):
                    arm_list.append(np.zeros((6,)))
            
            hybrid_feat_list = []
            for i in range(self.num_arms):
                x_i = np.outer(user_feat, arm_list[i]).reshape(-1)
                z_i = arm_list[i]
                hybrid_feat_list.append((x_i, z_i))

            return {'curr_arm': arm_id.index(a_t), 'curr_reward': r_t, 'arm_feats': hybrid_feat_list}
        
    # def add_to_cache(self):
    #     counter = 0
    #     temp_cache = []
    #     for name in self.file_names:
    #         with gzip.open(name, 'rt') as f:
    #             for line in f:
    #                 data_dict = self.process_line(line)
    #                 temp_cache.append(data_dict)
    #                 counter += 1
    #                 if counter == self.temp_cache_size:
    #                     while True:
    #                         self.lock.acquire()
    #                         if self.T == 0:
    #                             self.lock.release()
    #                             return
    #                         elif len(self.cache) >= self.temp_cache_size - self.temp_cache_size:
    #                             self.lock.release()
    #                             time.sleep(5)
    #                         else:
    #                             self.cache.extend(temp_cache)
    #                             self.lock.release()
    #                             temp_cache = []
    #                             counter = 0
    #                             break                        
                    

    # def step(self):
    #     while True:
    #         self.lock.acquire()
    #         if len(self.cache) > 0:
    #             data = self.cache.pop(0)
    #             self.T -= 1
    #             self.lock.release()
    #             return data
    #         else:
    #             self.lock.release()
    #             time.sleep(2)
        
    def step(self):
        for name in self.file_names:
            with gzip.open(name, 'rt') as f:
                for line in f:
                    data_dict = self.process_line(line)
                    yield data_dict



def check_min_max_arm_sizes(folder, data='Yahoo'):
    env = OffEval(folder, data, 20)
    T = 1
    for data in tqdm(env.step()):
        #display_bar.set_description(f'{T} lines processed')
        if T == 1000000:
            break
        T += 1
    print('Max arms:', env.max_arms)
    print('Min arms:', env.min_arms)

def offline_simulator(alg, env, max_time, id, folder, name):
    T = 0
    for data in tqdm(env.step(), position=(id+1)):
        a_t = alg.predict(data['arm_feats'])
        if a_t == data['curr_arm']:
            alg.update(data['arm_feats'], data['curr_reward'])
        T += 1
        if T == max_time:
            break
    alg.save_results(f'{name}_{max_time}', folder)



def prepare_algo_arr(algo_dict, T, d, k, L, delta=0.001):
    from algorithms.linear import DisLinUCB_Offline, HyLinUCB_Offline, OFUL_Offline, MHyLinUCB_Offline#, SupLinUCB_Offline
    algo_arr = []
    for key in algo_dict.keys():
        if key == 'DisLinUCB':
            lmbda = algo_dict[key]['lambda']
            algo_arr.append(DisLinUCB_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.25, lmbda))
        elif key == 'LinUCB':
            lmbda = algo_dict[key]['lambda']
            algo_arr.append(HyLinUCB_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.25, lmbda))
        elif key == 'OFUL':
            lmbda = algo_dict[key]['lambda']
            algo_arr.append(OFUL_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.25, lmbda))
        elif key == 'MHyLinUCB':
            lmbda = algo_dict[key]['lambda']
            algo_arr.append(MHyLinUCB_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.25, lmbda))
        # elif key == 'SupLinUCB':
        #     lmbda = algo_dict[key]['lambda']
        #     algo_arr.append(SupLinUCB_Offline(d, k, L, delta, 2.0, 1.0, 2.0, 1.0, 0.25, lmbda, T))
    return algo_arr
        

def plot_reward(infolder, dataname, timestep, folder=None, name=None, show=False):
    _, ax = plt.subplots(1, 1, figsize=(6, 4))
    for root, _, files in os.walk(infolder):
        for name1 in files:
            if dataname == name1.split('_')[0] and str(timestep) == name1.split('_')[1]:
                print(name1)
                df = pd.read_csv(os.path.join(root, name1))
                T = len(df)
                ax.plot(np.arange(1, T+1), np.cumsum(df['reward']), label=name1.split('_')[2])        
    
    ax.set_ylabel('CTR')
    ax.set_xlabel('Time')
    ax.legend()
    ax.grid()
    if name is not None:
        ax.set_title(name)
    if show:
        plt.show()
    else:
        assert folder is not None, "Folder must be given"
        plt.savefig(os.path.join(folder, f'{name}-CTR.png'), dpi=200)

def main(name, zipfolder, num_arms, timesteps, model, output):
        if name == 'Yahoo':
            d, k, L = 36, 6, int(num_arms)
        if model ==  'Linear':
            algo_dict = {'MHyLinUCB': {'lambda': 0.01},
                        'LinUCB': {'lambda': 0.01},
                        'DisLinUCB': {'lambda': 0.01},
                        'HyLinUCB': {'lambda': 0.01}}
        algo_arr = prepare_algo_arr(algo_dict, timesteps, d, k, L, delta=0.001)
        pool = Pool(processes=3)
        args = [(alg, OffEval(zipfolder, name, num_arms), timesteps, i+1, output, name)\
                for i, alg in enumerate(algo_arr)]
        pool.starmap(offline_simulator, args)
        pool.close()
        pool.join()
        plot_reward(output, name, timesteps, output, name)


if __name__=='__main__':
    # check_min_max_arm_sizes('./Dataset/Yahoo-Front-Page/R6')
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the dataset [options: Yahoo]', default='Yahoo')
    parser.add_argument('-z', '--zipfolder', type=str, help='path to zipped data folder')
    parser.add_argument('-L', '--num_arms', type=int, help='number of items', default=20)
    parser.add_argument('-T', '--timesteps', type=int, help='number of time steps', default=100000)
    parser.add_argument('-m', '--model', type=str, help='model type [options: Linear]', default='Linear')
    parser.add_argument('-o', '--output', type=str, help='output folder path', default='./Results')
    args = parser.parse_args()
    main(args.name, args.zipfolder, args.num_arms, args.timesteps, args.model, args.output)
    # elif args.model == 'Logistic':
    #     algo_dict = {'HyEcoLog': {'lambda': 1.0},
    #                 'DisEcoLog': {'lambda': 1.0},
    #                 'MHyEcoLog': {'lambda': 1.0}}




            
        
# python offline_evaluator.py -z ./Dataset/Yahoo-Front-Page/R6 -T 1000