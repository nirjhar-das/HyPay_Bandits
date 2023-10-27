import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithms import HyLinUCB
from environment import HybridBandits

def simulate(env, algo, T):
    for _ in tqdm(range(T)):
        a_t = algo.next_action()
        reward, regret = env.step(a_t)
        algo.update(reward, regret)
    #algo.save_results()


def main(env_name, config, num_trials, delta, lmbda, T, output_folder='.'):
    env = HybridBandits(env_name, config)
    all_rewards = np.zeros((num_trials, T))
    all_regrets = np.zeros((num_trials, T))
    for i in range(num_trials):
        algo = HyLinUCB(i+1, env.arms, delta, env.M, env.N, env.S1, env.S2, lmbda)
        print('Simulating Trial', i)
        simulate(env, algo, T)
        all_rewards[i] += np.array(algo.rewards)
        all_regrets[i] += np.array(algo.regrets)
    mean_reward = np.mean(all_rewards, axis=0)
    std_reward = np.std(all_rewards, axis=0)
    mean_regret = np.mean(all_regrets, axis=0)
    std_regret = np.std(all_regrets, axis=0)
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
    config['theta_dim'] = 10
    config['beta_dim'] = 2
    config['theta_norm'] = 2.0
    config['beta_norm'] = 1.0
    config['x_norm'] = 1.0
    config['z_norm'] = 1.0
    env_name = 'Testbench_1'
    main(env_name, config, num_trials=5, delta=0.01, lmbda=0.01, T=50000)

    
