import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

def main(input_folder, output_folder):
#    colors = plt.cm.get_cmap('hsv', 1)
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename[-4:] == '.csv':
                df = pd.read_csv(filename)
                idx_arr = [m.start() for m in re.finditer('_', filename)]
                algo_name = filename[idx_arr[-2]+1 : idx_arr[-1]]
            else:
                continue
            x = np.arange(1, len(df)+1)
            ax[0][0].plot(x, df['mean_reward'], label=algo_name)
            min_reward = np.array(df['mean_reward']) - np.array(df['std_reward'])
            max_reward = np.array(df['mean_reward']) + np.array(df['std_reward'])
            ax[0][0].fill_between(x, min_reward, max_reward, alpha=0.2)

            ax[0][1].plot(x, df['mean_regret'], label=algo_name)
            min_regret = np.array(df['mean_regret']) - np.array(df['std_regret'])
            max_regret = np.array(df['mean_regret']) + np.array(df['std_regret'])
            ax[0][1].fill_between(x, min_regret, max_regret, alpha=0.2)

            ax[1][0].plot(x, df['time_avg_reward'], label=algo_name)
            ax[1][1].plot(x, df['time_avg_regret'], label=algo_name)

    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlabel('Time Steps')
            ax[i][j].grid()
            ax[i][j].legend()
    
    ax[0][0].set_ylabel('Cumulative Reward')
    ax[0][1].set_ylabel('Cumulative Regret')
    ax[1][0].set_ylabel('Time Avg Reward')
    ax[1][1].set_ylabel('Time Avg Regret')

    plt.savefig(os.path.join(output_folder, 'Result.png'), dpi=600)
    plt.close()


if __name__ == '__main__':
    main('.', '.')