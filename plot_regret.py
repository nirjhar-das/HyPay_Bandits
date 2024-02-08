import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_time_vs_regret(folder):
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not file.startswith('Testbench'):
                continue
            name_arr = file.split('_')
            df = pd.read_csv(os.path.join(root, file))
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            T_arr = np.arange(1, len(df[df.columns[0]])+1)
            for col in df.columns:
                ax.plot(T_arr, df[col].cumsum(), label=col)
            ax.grid()
            ax.legend()
            filename = f'Reg_vs_T_{int(name_arr[1])}_{int(name_arr[2])}_{int(name_arr[3])}.svg'
            plt.savefig(os.path.join(root, filename), dpi=200, format='svg')


def plot_num_arms_vs_regret(folder):
    final_regret = {}
    idx = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not file.startswith('Testbench'):
                continue
            name_arr = file.split('_')
            if name_arr[1] == '10' and name_arr[2] == '10':
                df = pd.read_csv(os.path.join(root, file))
                ls = df.cumsum().iloc[-1]
                idx.append(int(name_arr[3]))
                if len(final_regret.keys()) == 0:
                    for k in ls.keys():
                        final_regret[k] = [ls[k]]
                else:
                    for k in ls.keys():
                        final_regret[k].append(ls[k])
        
        #final_df = pd.DataFrame(final_regret, index=idx)
        #final_df.sort_index(inplace=True)
        idx = np.array(idx)
        idx_sorted = np.argsort(idx)
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
        for k in final_regret.keys():
            ax.plot(idx[idx_sorted], np.array(final_regret[k])[idx_sorted], marker='o', markersize=8, linestyle='dashed', label=k)
        #ax = final_df.plot(kind='bar', rot=0)
        ax.grid()
        ax.legend()
        plt.savefig(os.path.join(folder, 'Reg_vs_L_bar_plot.svg'), dpi=200, format='svg')


if __name__ == '__main__':
    folder = 'Results'
    plot_num_arms_vs_regret(folder)
    plot_time_vs_regret(folder)
