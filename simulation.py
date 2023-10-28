import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithms import HyLinUCB, DisLinUCB
from environment import HybridBandits

def simulate(env, algo_arr, T):
    for _ in tqdm(range(T)):
        for algo in algo_arr:
            a_t = algo.next_action()
            reward, regret = env.step(a_t)
            algo.update(reward, regret)
    #algo.save_results()


def main(env_name, config, num_trials, delta, lmbda, T, output_folder='.'):
    env = HybridBandits(env_name, config)
    all_rewards = [np.zeros((num_trials, T)) for _ in range(2)]
    all_regrets = [np.zeros((num_trials, T)) for _ in range(2)]
    for i in range(num_trials):
        algo1 = HyLinUCB(i+1, env.arms, delta, env.M, env.N, env.S1, env.S2, lmbda)
        algo2 = DisLinUCB(i+1, env.arms, delta, env.M, env.N, env.S1, env.S2, lmbda)
        algo_arr = [algo1, algo2]
        print('Simulating Trial', i+1)
        simulate(env, algo_arr, T)
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
        df.to_csv(os.path.join(output_folder, \
                            f'{env_name}_{algo.name[:algo.name.index("_")]}_{num_trials}.csv')\
                , index=False)

    

if __name__ == '__main__':
    config = {}
    config['seed'] = np.random.randint(1098321)
    print('Seed:', config['seed'])
    config['model_type'] = 'Linear'
    config['num_labels'] = 50
    config['theta_dim'] = 20
    config['beta_dim'] = 4
    config['theta_norm'] = 5.0
    config['beta_norm'] = 3.0
    config['x_norm'] = 1.0
    config['z_norm'] = 1.0
    env_name = 'Testbench_1'
    main(env_name, config, num_trials=5, delta=0.01, lmbda=0.01, T=10000)

    
