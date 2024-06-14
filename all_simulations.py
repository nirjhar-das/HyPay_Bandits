import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from environment import HybridBandits
from algorithms.linear import DisLinUCB, LinUCB, OFUL, SupLinUCB, HyLinUCB, HyRan


def simulate_linear(env, algo_arr, T, i):
    for t in tqdm(range(T), position=i):
        a_t = []
        for algo in algo_arr:
            a = algo.next_action()
            a_t.append(a)
        rewards_t, regrets_t, action_set_t = env.step(a_t)
        for i, algo in enumerate(algo_arr):
            algo.update(rewards_t[i], regrets_t[i], action_set_t)
    return algo_arr


def multi_simulation_linear(num_trials, algo_dict, env:HybridBandits, delta:float, T:int):
    all_rewards = [np.zeros((num_trials, T)) for _ in range(len(algo_dict.keys()))]
    all_regrets = [np.zeros((num_trials, T)) for _ in range(len(algo_dict.keys()))]
    args_arr = []
    for i in range(num_trials):
        copy_env = HybridBandits(copy_env=env, seed=i)
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
            elif k == 'SupLinUCB':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(SupLinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, env.sigma, lmbda, T))
            elif k == 'HyRan':
                p = algo_dict[k]['p']
                algo_arr.append(HyRan(env.get_first_action_set(), p))
        print('Simulating Trial', i+1)
        args_arr.append((copy_env, algo_arr, T, i+1))
    with Pool() as p:
        algos_arr_arr = p.starmap(simulate_linear, args_arr)

    for algo_arr_val in algos_arr_arr:    
        for j in range(len(algo_arr_val)):
            all_rewards[j][i] += np.array(algo_arr_val[j].rewards)
            all_regrets[j][i] += np.array(algo_arr_val[j].regrets)
    return all_rewards, all_regrets

def all_simulations(d, k, L, T, name, model_type, num_trials, num_envs, folder, seed=194821263):
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
    delta = 0.01
    
    if (d == 40 and k == 5 and L == 25) or (d == 5 and k == 40 and L == 25):
        all_rewards = [np.zeros((T,)) for _ in range(len(['HyLinUCB', 'DisLinUCB', 'LinUCB', 'HyRan', 'SupLinUCB']))]
        all_regrets = [np.zeros((T,)) for _ in range(len(['HyLinUCB', 'DisLinUCB', 'LinUCB', 'HyRan', 'SupLinUCB']))]
        # all_rewards = [np.zeros((T,)) for _ in range(len(['HyLinUCB', 'DisLinUCB', 'LinUCB', 'HyRan']))]
        # all_regrets = [np.zeros((T,)) for _ in range(len(['HyLinUCB', 'DisLinUCB', 'LinUCB', 'HyRan']))]
    else:
        all_rewards = [np.zeros((T,)) for _ in range(len(['HyLinUCB', 'DisLinUCB', 'LinUCB', 'HyRan']))]
        all_regrets = [np.zeros((T,)) for _ in range(len(['HyLinUCB', 'DisLinUCB', 'LinUCB', 'HyRan']))]
    print(f'Configuaration: d={d}, k={k}, L={L}, T={T}, type={model_type}')
    for i in range(num_envs):
        print('Simulating Env ', i+1, ' of ', num_envs)
        config['seed'] = rng.integers(1074926307) # Uncomment the random seed generator for random instances
        # Name of the simulation
        env = HybridBandits(name, config)
        if model_type ==  'Linear':
            if (d == 40 and k == 5 and L == 25) or (d == 5 and k == 40 and L == 25):
                algo_dict = {'HyLinUCB': {'lambda': 0.1},
                        'LinUCB': {'lambda': 0.1},
                        'DisLinUCB': {'lambda': 0.1},
                        'SupLinUCB': {'lambda': 0.1},
                        'HyRan': {'lambda': 1.0, 'p': 0.5}}
            else:
                algo_dict = {'HyLinUCB': {'lambda': 0.1},
                            'LinUCB': {'lambda': 0.1},
                            'DisLinUCB': {'lambda': 0.1},
                            #'SupLinUCB': {'lambda': 0.1},
                            'HyRan': {'lambda': 1.0, 'p': 0.5}}
            rewards, regrets = multi_simulation_linear(num_trials, algo_dict, env, delta, T)
            regrets_dict = {k2: np.sum(regrets[i], axis=0) / num_trials for i, k2 in enumerate(algo_dict.keys())}
            df = pd.DataFrame(data=regrets_dict)
            filename = f'{name}_Trial_{i+1}_{d}_{k}_{L}_{T}_{model_type}.csv'
            df.to_csv(os.path.join(folder, filename), index=False)

        for j in range(len(algo_dict.keys())):
            for k1 in range(num_trials):
                all_rewards[j] += rewards[j][k1]
                all_regrets[j] += regrets[j][k1]
    
    
    for j in range(len(algo_dict.keys())):
        all_rewards[j] /= (num_envs * num_trials)
        all_regrets[j] /= (num_envs * num_trials)
    
    result_dict = {c : all_regrets[s] for s, c in enumerate(algo_dict.keys())}
    filename = f'{name}_{d}_{k}_{L}_{T}_{model_type}.csv'
    df = pd.DataFrame(data=result_dict)
    df.to_csv(os.path.join(folder, filename), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '-m', type=str, default='Linear', help='Model Type')
    parser.add_argument('--name', '-n', type=str, default='Testbench', help='Name of Experiment')
    parser.add_argument('--output', '-o', type=str, default='./Results', help='Output folder to store')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if args.model_type == 'Linear':
        d_arr = [5, 40]
        k_arr = [40, 5]
        L_arr = [25] + [10, 50, 100, 200, 300, 400]
        T = 30000
        for k in k_arr:
            for d in d_arr:
                for  L in L_arr:
                    if(d == 5 and k == 40 and L == 25):
                        all_simulations(d, k, L, 80000, args.name, 'Linear', 5, 5, args.output)
                    elif(d == 40 and k == 5 and L == 25):
                        all_simulations(d, k, L, 80000, args.name, 'Linear', 5, 5, args.output)
                    elif(d == 5 and k == 5):
                        all_simulations(d, k, L, T, args.name, 'Linear', 5, 5, args.output)
