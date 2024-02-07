import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(folder):
    fig, ax = plt.subplots(1, 1)
    L, k, num_trials = 0, 0, 0
    x = []
    R = {'LinUCB': [], 'DisLinUCB': []}
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            dim = dir.split('_')[-1]
            x.append(int(dim))
            for subroot, _, subfiles in os.walk(os.path.join(root, dir)):
                for subfile in subfiles:
                    if subfile[-4:] == '.csv':
                        df = pd.read_csv(os.path.join(subroot, subfile))
                        idx_arr = [m.start() for m in re.finditer('_', subfile)]
                        algo_name = subfile[idx_arr[-2]+1 : idx_arr[-1]]
                        R[algo_name].append(df['mean_regret'].iloc[-1])
                        if num_trials == 0:
                            num_trials = int(subfile[idx_arr[-1]+1:subfile.index('.')])
                    elif (L == 0) and (subfile[-5:] == '.json'):
                        with open(os.path.join(subroot, subfile), 'r') as f:
                            env_dict = json.load(f)
                            L = env_dict['L']
                            k = env_dict['k']
    
    idx = np.argsort(x)
    x = [x[i] for i in idx]
    R['DisLinUCB'] = [R['DisLinUCB'][i] for i in idx]
    R['LinUCB']  = [R['LinUCB'][i] for i in idx]
    ax.plot(x, R['DisLinUCB'], label='DisLinUCB')
    ax.plot(x, R['LinUCB'], label='LinUCB')
    ax.grid()
    ax.legend()
    ax.set_xlabel(f'Theta dimension for L = {L} and beta_dim = {k}')
    ax.set_ylabel('Avg Regret over 5 trials')

    plt.savefig(os.path.join(folder, 'Result.png'), dpi=200)
    plt.close()


if __name__ == '__main__':
    main('./D_Comparison')