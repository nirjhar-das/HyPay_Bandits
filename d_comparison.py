import os
import numpy as np
import pandas as pd
import simulation

def main(init_d, incr, max_d=None, out_path='.'):
    config = {}
    config['seed'] = np.random.randint(1098321)
    config['model_type'] = 'Linear'
    config['horizon_length'] = 20000
    config['num_labels'] = 10
    config['num_context'] = 10
    config['theta_dim'] = init_d
    config['beta_dim'] = 2
    config['theta_norm'] = 1.0
    config['beta_norm'] = 0.1
    config['x_norm'] = 1.0
    config['z_norm'] = 1.0
    config['is_easy'] = False
    folder = out_path
    if not os.path.exists(folder):
        os.mkdir(folder)
    if max_d is None:
        max_d = config['num_labels']*config['beta_dim']
        d = init_d
        while(d <= config['num_labels']*config['beta_dim']):
            env_name = f'Test_{d}'
            print('Simulation for theta dimension =', d)
            simulation.main(env_name, config, 5, 0.001, 0.0005, folder, normalize_regret=True)
            d += incr
            config['theta_dim'] = d


if __name__ == '__main__':
    out_folder = './D_Comparison'
    main(2, 2, out_path=out_folder)