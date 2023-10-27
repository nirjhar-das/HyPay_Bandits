import numpy as np
from tqdm import tqdm
from .algorithms import HyLinUCB
from environment import HybridBandits

def simulate(env, algo, T):
    for _ in tqdm(range(T)):
        a_t = algo.next_action()
        reward, regret = env.step(a_t)
        algo.update(reward, regret)
    algo.save_results()


def main(env_name, config, num_trials, delta, lmbda, T):
    env = HybridBandits(env_name, config)
    for i in range(num_trials):
        algo = HyLinUCB(i+1, env.arms, delta, env.M, env.N, env.S1, env.S2, lmbda)
        print('Simulating Trial', i)
        simulate(env, algo, T)
    


if __name__ == '__main__':
    config = {}
    config['seed'] = np.random.randint(10726498321)
    print('Seed:', config['seed'])
    config['model_type'] = 'Linear'
    config['num_labels'] = 200
    config['theta_dim'] = 20
    config['beta_dim'] = 2
    config['theta_norm'] = 10.0
    config['beta_norm'] = 2.0
    config['x_norm'] = 1.0
    config['z_norm'] = 1.0
    env_name = 'Testbench_1'
    main(env_name, config, num_trials=5, delta=0.01, lmbda=5.0, T=20000)

    
