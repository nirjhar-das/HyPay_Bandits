import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithms import LinUCB, DisLinUCB, LinUCBLangford, HyLinUCBv2
from environment import HybridBandits
from simulation import simulate

def single_trial_algo_gridsearch(algo_names, hyperparam_grid, env):
    algo_list = []
    for algo_name in algo_names:
        if algo_name == 'LinUCBLangford':
            for alpha in hyperparam_grid['alpha']:
                algo = LinUCBLangford(env.get_first_action_set(), env.M, env.N, env.S1, env.S2, alpha, info=alpha)
                algo_list.append(algo)
        else:
            delta = hyperparam_grid['delta']
            for lmbda in hyperparam_grid['lambda']:
                for gamma in hyperparam_grid['gamma']:
                    if algo_name == 'LinUCB':
                        algo = LinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda, gamma, info=f'{lmbda}_{gamma}')
                    if algo_name == 'HyLinUCBv2':
                        algo = HyLinUCBv2(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda, gamma, info=f'{lmbda}_{gamma}')
                    if algo_name == 'DisLinUCB':
                        algo = DisLinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda, info=f'{lmbda}_{gamma}')
                    algo_list.append(algo)
    m = simulate(env, algo_list, env.T)
    return {algo.name: algo.regrets[-1] for algo in algo_list}

def multi_trial_avg_performance(algo_names, hyperparam_grid, env, num_trials):
    tot_regret = None
    for i in range(num_trials):
        if tot_regret is None:
            tot_regret = single_trial_algo_gridsearch(algo_names, hyperparam_grid, env)
        else:
            d = single_trial_algo_gridsearch(algo_names, hyperparam_grid, env)
            for k in tot_regret.keys():
                tot_regret[k] += d[k]
        env.reset()
    
    best_hyperparam_dict = {'delta': hyperparam_grid['delta']}
    for algo_name in algo_names:
        m, k_m = np.inf, None
        for k in tot_regret.keys():
            if (k.startswith(algo_name)) and (tot_regret[k] < m):
                m = tot_regret[k]
                k_m = k
        arr = k_m.split('_')
        if algo_name == 'LinUCBLangford':
            best_hyperparam_dict[algo_name] = {'alpha': float(arr[1])}
        else:
            best_hyperparam_dict[algo_name] = {'lambda': float(arr[1]), 'gamma': float(arr[2])}
    
    return best_hyperparam_dict

def main(env_load_path, algo_names, hyperparam_grid, num_trials):
    env = HybridBandits(load=env_load_path)
    d = multi_trial_avg_performance(algo_names, hyperparam_grid, env, num_trials)
    with open(os.path.join(os.path.dirname(env_load_path), 'Best_Hyperparam.json'), 'w+') as f:
        json.dump(d, f, indent=4)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadpath', type=str, required=True)
    args = parser.parse_args()
    algo_names = ['LinUCBLangford', 'LinUCB', 'HyLinUCBv2']
    hyper_param_grid = {'delta': 0.001, 'alpha': [0.0005, 0.001],\
                        'lambda': [0.0005, 0.001], \
                        'gamma': [0.0005, 0.001]}
    num_trials = 5
    main(args.loadpath, algo_names, hyper_param_grid, num_trials)