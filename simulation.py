import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithms import HyLinUCB, DisLinUCB
from environment import HybridBandits

def simulate(env, algo_arr, T):
    for _ in tqdm(range(T)):
        a_t = []
        for algo in algo_arr:
            a_t.append(algo.next_action())
        rewards_t, regrets_t, action_set_t = env.step(a_t)
        for i, algo in enumerate(algo_arr):
            algo.update(rewards_t[i], regrets_t[i], action_set_t)
    #algo.save_results()


def main(env_name, config, num_trials, delta, lmbda, T, output_folder='.'):
    out_folder = os.path.join(output_folder, env_name)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    env = HybridBandits(env_name, config)
    env.save_metadata(out_folder)
    all_rewards = [np.zeros((num_trials, T)) for _ in range(2)]
    all_regrets = [np.zeros((num_trials, T)) for _ in range(2)]
    for i in range(num_trials):
        algo1 = HyLinUCB(i+1, env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda)
        algo2 = DisLinUCB(i+1, env.get_first_action_set(), delta, env.M, env.N, env.S1, env.S2, lmbda)
        algo_arr = [algo1, algo2]
        print('Simulating Trial', i+1)
        simulate(env, algo_arr, T)
        env.reset()
        for j in range(len(algo_arr)):
            all_rewards[j][i] += np.array(algo_arr[j].rewards)
            all_regrets[j][i] += np.array(algo_arr[j].regrets)
    for i, algo in enumerate(algo_arr):
        mean_reward = np.mean(all_rewards[i], axis=0)
        std_reward = np.std(all_rewards[i], axis=0)
        mean_regret = np.mean(all_regrets[i], axis=0)
        std_regret = np.std(all_regrets[i], axis=0)
        rewards = np.cumsum(mean_reward)
        regrets = np.cumsum(mean_regret)
        time_avg_rewards = rewards / np.arange(1, T+1)
        time_avg_regrets = regrets / np.arange(1, T+1)
        results = {'mean_reward': rewards, 'mean_regret': regrets, 'std_reward': std_reward,\
                'std_regret': std_regret,\
                'time_avg_reward': time_avg_rewards, 'time_avg_regret': time_avg_regrets}
        df = pd.DataFrame(data=results)
        df.to_csv(os.path.join(out_folder, \
                            f'{env_name}_{algo.name[:algo.name.index("_")]}_{num_trials}.csv')\
                , index=False)

    

if __name__ == '__main__':
    config = {}
    config['seed'] = np.random.randint(1098321)
    print('Seed:', config['seed'])
    config['model_type'] = 'Linear'
    config['horizon_length'] = 100000
    config['num_labels'] = 10
    config['num_context'] = 10
    config['theta_dim'] = 5
    config['beta_dim'] = 2
    config['theta_norm'] = 1.0
    config['beta_norm'] = 0.1
    config['x_norm'] = 1.0
    config['z_norm'] = 1.0
    config['is_easy'] = False
    env_name = 'Testbench_3'
    folder = '.'
    main(env_name, config, num_trials=10, delta=0.001, lmbda=0.0005, T=config['horizon_length'], output_folder=folder)

    
