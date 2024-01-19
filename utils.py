import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_reward_plot(env, nrows, ncols):
    if nrows <= 1:
        fig, ax = plt.subplots(ncols, sharex=True, sharey=True)
        x = 0
        base = np.arange(1, env.L + 1)
        theta = env.parameters['theta']
        beta_arr = env.parameters['beta']
        for i in range(env.num_context):
            reward_arr = []
            for j, (a,b) in enumerate(env.arms[i]):
                if env.model_type == 'Linear':
                    reward_arr.append(np.dot(theta, a) + np.dot(beta_arr[j], b))
                else:
                    reward_arr.append(1.0 / (1.0 + np.exp(-np.dot(theta, a) - np.dot(beta_arr[j], b))))
            ax[x].bar(base, reward_arr)
            ax[x].grid()
            x += 1
        plt.show()
    else:
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        x, y = 0, 0
        base = np.arange(1, env.L + 1)
        theta = env.parameters['theta']
        beta_arr = env.parameters['beta']
        for i in range(env.num_context):
            if y == ncols:
                x += 1
                y = 0
            reward_arr = []
            for j, (a,b) in enumerate(env.arms[i]):
                if env.model_type == 'Linear':
                    reward_arr.append(np.dot(theta, a) + np.dot(beta_arr[j], b))
                else:
                    reward_arr.append(1.0 / (1.0 + np.exp(-np.dot(theta, a) - np.dot(beta_arr[j], b))))
            ax[x][y].bar(base, reward_arr)
            ax[x][y].grid()
            y += 1
        plt.show()


def plot_regret(result_dict, T):
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    x = np.arange(1, T+1)
    for k in result_dict.keys():
        ax[0][0].plot(x, result_dict[k]['mean_reward'], label=k)
        min_reward = np.array(result_dict[k]['mean_reward']) - np.array(result_dict[k]['std_reward'])
        max_reward = np.array(result_dict[k]['mean_reward']) + np.array(result_dict[k]['std_reward'])
        ax[0][0].fill_between(x, min_reward, max_reward, alpha=0.2)

        ax[0][1].plot(x, result_dict[k]['mean_regret'], label=k)
        min_regret = np.array(result_dict[k]['mean_regret']) - np.array(result_dict[k]['std_regret'])
        max_regret = np.array(result_dict[k]['mean_regret']) + np.array(result_dict[k]['std_regret'])
        ax[0][1].fill_between(x, min_regret, max_regret, alpha=0.2)

        ax[1][0].plot(x, result_dict[k]['time_avg_reward'], label=k)
        ax[1][1].plot(x, result_dict[k]['time_avg_regret'], label=k)

    for i in range(2):
        for j in range(2):
            ax[i][j].set_xlabel('Time Steps')
            ax[i][j].grid()
            ax[i][j].legend()
    
    ax[0][0].set_ylabel('Cumulative Reward')
    ax[0][1].set_ylabel('Cumulative Regret')
    ax[1][0].set_ylabel('Time Avg Reward')
    ax[1][1].set_ylabel('Time Avg Regret')
    
    plt.show()


def create_result_dict(all_rewards, all_regrets, algo_dict, T):
        result_dict = {}
        for i, algo in enumerate(algo_dict.keys()):
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
                result_dict[algo] = results
        return result_dict

def plot_action_freq(ls, env, nrows, ncols):
        if ncols <= 1:
            fig, ax = plt.subplots(nrows, sharex=False, sharey=False, figsize=(30, 30))
            x = 0
            base = [str(i) for i in np.arange(1, env.L + 1)]
            for i in range(env.num_context):
                act_freq_dict = ls[i]
                data = {}
                base_copy = [b for b in base]
                for k in act_freq_dict.keys():
                    data[k] = []
                    for a in range(env.L):
                        freq = np.sum(np.array(act_freq_dict[k]) == a)
                        if a == env.best_arm[i] and not(base_copy[a].endswith('*')):
                            base_copy[a] += '*'
                        data[k].append(freq)
                df = pd.DataFrame(data, index=base_copy)
                df.plot.bar(rot=0, width=0.8, ax=ax[x])
                ax[x].legend().set_visible(False)
                ax[x].set_title(f'Context:{i}')
                ax[x].grid()
                x += 1
            handles, labels = ax[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncols=2)
            plt.show()
        else:
            fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(30, 30))
            x, y = 0, 0
            base = [str(i) for i in np.arange(1, env.L + 1)]
            for i in range(env.num_context):
                if y == ncols:
                    x += 1
                    y = 0
                act_freq_dict = ls[i]
                data = {}
                base_copy = [b for b in base]
                for k in act_freq_dict.keys():
                    data[k] = []
                    for a in range(env.L):
                        freq = np.sum(np.array(act_freq_dict[k]) == a)
                        if a == env.best_arm[i] and not(base_copy[a].endswith('*')):
                            base_copy[a] += '*'
                        data[k].append(freq)
                df = pd.DataFrame(data, index=base_copy)
                df.plot.bar(rot=0, width=0.8, ax=ax[x][y])
                ax[x][y].legend().set_visible(False)
                ax[x][y].set_title(f'Context:{i}')
                ax[x][y].grid()
                y += 1
            handles, labels = ax[-1][-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncols=2)
            plt.show()