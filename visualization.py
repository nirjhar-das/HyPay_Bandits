import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def main(input_file, output_folder):
    colors = plt.cm.get_cmap('hsv', 1)
    df = pd.read_csv(input_file)
    x = np.arange(1, len(df)+1)
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    ax[0][0].plot(x, df['mean_reward'], color=colors(0))
    min_reward = np.array(df['mean_reward']) - np.array(df['std_reward'])
    max_reward = np.array(df['mean_reward']) + np.array(df['std_reward'])
    ax[0][0].fill_between(x, min_reward, max_reward, facecolor=colors(0), alpha=0.2)

    ax[0][1].plot(x, df['mean_regret'], color=colors(0))
    min_regret = np.array(df['mean_regret']) - np.array(df['std_regret'])
    max_regret = np.array(df['mean_regret']) + np.array(df['std_regret'])
    ax[0][1].fill_between(x, min_regret, max_regret, facecolor=colors(0), alpha=0.2)

    ax[1][0].plot(x, df['time_avg_reward'], color=colors(0))
    ax[1][1].plot(x, df['time_avg_regret'], color=colors(0))

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
    main('Testbench_1_HyLinUCB_1.csv', '.')