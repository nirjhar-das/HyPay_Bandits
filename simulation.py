import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithms import HyLinUCB, DisLinUCB, LinUCBLangford, HyLinUCBv2
from environment import HybridBandits
from plot_rewards import create_plot

def simulate(env, algo_arr, T):
    max_rew = 0
    for _ in tqdm(range(T)):
        a_t = []
        for algo in algo_arr:
            a_t.append(algo.next_action())
        max_rew += env.get_max_reward()
        rewards_t, regrets_t, action_set_t = env.step(a_t)
        for i, algo in enumerate(algo_arr):
            algo.update(rewards_t[i], regrets_t[i], action_set_t)
    return max_rew
    #algo.save_results()


def main(env, num_trials, delta, algo_dict, output_folder='.', normalize_regret=False):
    T = env.T
    all_rewards = [np.zeros((num_trials, T)) for _ in range(algo_dict.keys())]
    all_regrets = [np.zeros((num_trials, T)) for _ in range(algo_dict.keys())]
    for i in range(num_trials):
        algo_arr = []
        for k in algo_dict.keys():
            if k == 'HyLinUCB':
                lmbda = algo_dict[k]['lambda']
                gamma = algo_dict[k]['gamma']
                algo_arr.append(HyLinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda, gamma))
            elif k == 'DisLinUCB':
                lmbda = algo_dict[k]['lambda']
                algo_arr.append(DisLinUCB(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda))
            elif k == 'LinUCBLangford':
                alpha = algo_dict[k]['alpha']
                algo_arr.append(LinUCBLangford(env.get_first_action_set(), env.M, env.N, env.S1, env.S2, alpha))
            elif k == 'HyLinUCBv2':
                lmbda = algo_dict[k]['lambda']
                gamma = algo_dict[k]['gamma']
                algo_arr.append(HyLinUCBv2(env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda, gamma))
        print('Simulating Trial', i+1)
        m = simulate(env, algo_arr, T)
        env.reset()
        for j in range(len(algo_arr)):
            all_rewards[j][i] += np.array(algo_arr[j].rewards)
            all_regrets[j][i] += np.array(algo_arr[j].regrets)
    for i, algo in enumerate(algo_arr):
        mean_reward = np.mean(all_rewards[i], axis=0)
        std_reward = np.std(all_rewards[i], axis=0)
        if not normalize_regret:
            m = 1.0
        mean_regret = np.mean(all_regrets[i] / m, axis=0)
        std_regret = np.std(all_regrets[i] / m, axis=0)
        rewards = np.cumsum(mean_reward)
        regrets = np.cumsum(mean_regret)
        time_avg_rewards = rewards / np.arange(1, T+1)
        time_avg_regrets = regrets / np.arange(1, T+1)
        results = {'mean_reward': rewards, 'mean_regret': regrets, 'std_reward': std_reward,\
                'std_regret': std_regret,\
                'time_avg_reward': time_avg_rewards, 'time_avg_regret': time_avg_regrets}
        df = pd.DataFrame(data=results)
        df.to_csv(os.path.join(output_folder, \
                            f'{algo.name}_{num_trials}.csv')\
                , index=False)
        nrows = env.num_context // 5
        create_plot(env, output_folder, nrows, 5)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadpath', type=str, help='path to load environment from')
    parser.add_argument('-o', '--output', type=str, help='output folder path')
    parser.add_argument('-n', '--ntrials', type=int, help='number of trials', default=2)
    parser.add_argument('-t', '--testname', type=str, required=True)
    args = parser.parse_args()
    if args.loadpath is None:
        config = {}
        config['seed'] = np.random.randint(1098321)
        print('Seed:', config['seed'])
        config['model_type'] = 'Linear'
        config['horizon_length'] = 50000
        config['num_labels'] = 20
        config['num_context'] = 10
        config['theta_dim'] = 5
        config['beta_dim'] = 5
        config['theta_norm'] = 0.5
        config['beta_norm'] = 0.5
        config['x_norm'] = 1.0
        config['z_norm'] = 1.0
        config['is_easy'] = False
        env_name = args.testname
        folder = '.'
        if args.output is None:
            out_folder = os.path.join(folder, env_name)
        else:
            out_folder = args.output
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        env = HybridBandits(env_name, config)
        env.save_metadata(out_folder)
    else:
        env = HybridBandits(load=args.loadpath)
        print('Loaded the environment')
        out_folder = args.output
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
    main(env, num_trials=args.ntrials, delta=0.001, alpha=0.1, lmbda=0.1, gamma=0.1, output_folder=out_folder)

    
