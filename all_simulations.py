import os
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

from environment import HybridBandits
from algorithms.linear import DisLinUCB, LinUCB, OFUL, MHyLinUCB, SupLinUCB, HyLinUCB, HyRan
from algorithms.logistic import HyEcoLog, DisEcoLog

def simulate_linear(env, algo_arr, T):
    for t in tqdm(range(T)):
        a_t = []
        for algo in algo_arr:
            a = algo.next_action()
            a_t.append(a)
        rewards_t, regrets_t, action_set_t = env.step(a_t)
        for i, algo in enumerate(algo_arr):
            algo.update(rewards_t[i], regrets_t[i], action_set_t)

def simulate_logistic(env, algo_arr, T):
    for t in tqdm(range(T)):
        a_t = []
        for algo in algo_arr:
            a = algo.next_action()
            a_t.append(a)
        rewards_t, regrets_t, action_set_t = env.step(a_t)
        for i, algo in enumerate(algo_arr):
            algo.update(rewards_t[i], regrets_t[i], action_set_t)


def multi_simulation_linear(num_trials, algo_dict, env:HybridBandits, delta:float, T:int):
    all_rewards = [np.zeros((num_trials, T)) for _ in range(len(algo_dict.keys()))]
    all_regrets = [np.zeros((num_trials, T)) for _ in range(len(algo_dict.keys()))]
    for i in range(num_trials):
        algo_arr = []
        for k in algo_dict.keys():
            if k == 'DisLinUCB':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(DisLinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, env.sigma, lmbda))
            elif k == 'LinUCB':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(LinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, env.sigma, lmbda))
            elif k == 'OFUL':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(OFUL(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, env.sigma, lmbda))
            elif k == 'HyLinUCB':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(HyLinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, env.sigma, lmbda))
            elif k == 'MHyLinUCB':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(MHyLinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, env.sigma, lmbda))
            elif k == 'SupLinUCB':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(SupLinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, env.sigma, lmbda, T))
            elif k == 'HyRan':
                lmbda = algo_dict[k]['lambda']
                p = algo_dict[k]['p']
                algo_arr.append(HyRan(env.get_first_action_set(), lmbda, p))
        print('Simulating Trial', i+1)
        simulate_linear(env, algo_arr, T)
        env.reset()
        for j in range(len(algo_arr)):
            all_rewards[j][i] += np.array(algo_arr[j].rewards)
            all_regrets[j][i] += np.array(algo_arr[j].regrets)
    return all_rewards, all_regrets

def multi_simulation_logistic(num_trials, algo_dict, env:HybridBandits, delta:float, T:int):
    all_rewards = [np.zeros((num_trials, T)) for _ in range(len(algo_dict.keys()))]
    all_regrets = [np.zeros((num_trials, T)) for _ in range(len(algo_dict.keys()))]
    for i in range(num_trials):
        algo_arr = []
        for k in algo_dict.keys():
            if k == 'DisEcoLog':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(DisEcoLog(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda, env.kappa))
            elif k == 'HyEcoLog':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(HyEcoLog(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda, env.kappa))
            elif k == 'MHyEcoLog':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(HyEcoLog(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda, env.kappa))
        print('Simulating Trial', i+1)
        simulate_logistic(env, algo_arr, T)
        env.reset()
        for j in range(len(algo_arr)):
            all_rewards[j][i] += np.array(algo_arr[j].rewards)
            all_regrets[j][i] += np.array(algo_arr[j].regrets)
    return all_rewards, all_regrets

def all_simulations(d, k, L, T, model_type, num_trials, num_envs, seed=194821263):
    rng = np.random.default_rng(seed)
    config = {}
    config['model_type'] = model_type
    config['horizon_length'] = T       # Number of time steps T
    config['num_labels'] = L               # Number of actions L
    config['theta_dim'] = d                 # Dimension of theta d
    config['beta_dim'] = k                  # Dimension of beta k
    config['theta_norm'] = 0.8              # Max norm of theta M
    config['beta_norm'] = 0.5               # Max norm of beta_i's N
    config['x_norm'] = 1.0                  # Max norm of x
    config['z_norm'] = 1.0                  # Max norm of z
    config['subgaussian'] = 0.01             # Subgaussianity of noise
    delta = 0.001
    print(f'Configuaration: d={d}, k={k}, L={L}, T={T}, type={model_type}')
    for i in range(num_envs):
        print('Simulating Env ', i+1, ' of ', num_envs)
        config['seed'] = rng.integers(1074926307) # Uncomment the random seed generator for random instances
        env_name = 'Testbench'                 # Name of the simulation
        env = HybridBandits(env_name, config)
        if model_type ==  'Linear':
            if (d == 100 and k == 10 and L == 25) or (d == 10 and k == 100 and L == 25):
                # algo_dict = {'HyLinUCB': {'lambda': 0.01},
                #         'LinUCB': {'lambda': 0.01},
                #         'DisLinUCB': {'lambda': 0.01},
                #         'SupLinUCB': {'lambda': 0.01},
                #         'HyRan': {'lambda': 1.0, 'p': 0.65}}
                algo_dict = {'HyRan': {'lambda': 1.0, 'p': 0.5}}
            else:
                # algo_dict = {'HyLinUCB': {'lambda': 0.01},
                #             'LinUCB': {'lambda': 0.01},
                #             'DisLinUCB': {'lambda': 0.01}}
                algo_dict = {'HyRan': {'lambda': 1.0, 'p': 0.5}}
            rewards, regrets = multi_simulation_linear(num_trials, algo_dict, env, delta, T)
        elif model_type == 'Logistic':
            algo_dict = {'HyEcoLog': {'lambda': 1.0},
                        'DisEcoLog': {'lambda': 1.0},
                        'MHyEcoLog': {'lambda': 1.0}}
            rewards, regrets = multi_simulation_logistic(num_trials, algo_dict, env, delta, T)
        all_rewards = [np.zeros((T,)) for _ in range(len(algo_dict.keys()))]
        all_regrets = [np.zeros((T,)) for _ in range(len(algo_dict.keys()))]
        for j in range(len(algo_dict.keys())):
            for k1 in range(num_trials):
                all_rewards[j] += rewards[j][k1]
                all_regrets[j] += regrets[j][k1]
    
    
    for j in range(len(algo_dict.keys())):
        all_rewards[j] /= (num_envs * num_trials)
        all_regrets[j] /= (num_envs * num_trials)
    
    result_dict = {c : all_regrets[s] for s, c in enumerate(algo_dict.keys())}
    filename = f'{env_name}_{d}_{k}_{L}_{T}_{model_type}_HyRan.csv'
    df = pd.DataFrame(data=result_dict)
    df.to_csv(os.path.join('Results', filename), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '-m', type=str, required=True, help='Model Type')
    args = parser.parse_args()
    if args.model_type == 'Linear':
        #d_arr = [100, 10]
        d_arr = [10, 100]
        k_arr  = [10, 100]
        #L_arr = [25] + [2**i for i in range(1, 11)]
        L_arr = [25] + [100, 200, 300]
        T = 10000
        for k in k_arr:
            for d in d_arr:
                for  L in L_arr:
                    if(d == 10 and k == 100 and L == 25):
                        all_simulations(d, k, L, T, 'Linear', 3, 3)
                    elif(d == 100 and k == 10 and L == 25):
                        all_simulations(d, k, L, T, 'Linear', 3, 3)
                    elif(d == 10 and k == 10 and L != 25):
                        all_simulations(d, k, L, T, 'Linear', 2, 2)
    elif args.model_type == 'Logistic':
        T = 2000
        d_arr = [0, 3, 10, 16]
        k_arr = [0, 3, 5]
        L_arr = [10, 20]
        for L in L_arr:
            for  d in d_arr:
                for  k in k_arr:
                    if(((d == 0) and (k != 6)  and (L != 20)) or ((d != 10) and (k == 0) and (L != 20))):
                        continue
                    all_simulations(d, k, L, T, 'Logistic', 5, 5)