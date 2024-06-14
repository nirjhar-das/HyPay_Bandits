import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import get_color

def plot_time_vs_regret(folder, name, id):
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if (not file.startswith(name)) or (f'Trial_{id}' not in file):
                continue
            name_arr = file.split('_')
            df = pd.read_csv(os.path.join(root, file))
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            T_arr = np.arange(1, len(df[df.columns[0]])+1)
            for col in df.columns:
                if col == 'SupLinUCB': # SupLinUCB not plotted
                    continue
                ax.plot(T_arr, df[col].cumsum(), label=col, color=get_color(col))
            ax.grid()
            ax.legend(fontsize=15)
            ax.set_title(f'$ d_1 $ = {int(name_arr[3])}, $ d_2 $ = {int(name_arr[4])}, K = {int(name_arr[5])}', fontsize=20)
            ax.set_xlabel('Time', size=20)
            ax.set_ylabel('Regret', size=20)
            filename = f'Reg_vs_T_{int(name_arr[3])}_{int(name_arr[4])}_{int(name_arr[5])}_{id}.png'
            plt.savefig(os.path.join(root, filename), dpi=200, format='png')
            plt.close()

def plot_time_vs_regret_avg(folder, name):
    
    for root, _, files in os.walk(folder):
        for file in files:
            if (not file.startswith(name)) or ('Trial' in file) or ('80000' not in file):
                continue
            name_arr = file.split('_')
            df = pd.read_csv(os.path.join(root, file))
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            T_arr = np.arange(1, len(df[df.columns[0]])+1)
            for col in df.columns:
                if col == 'SupLinUCB': # SupLinUCB not plotted
                    continue
                ax.plot(T_arr, df[col].cumsum(), label=col, color=get_color(col))
            ax.grid()
            ax.legend(fontsize=15)
            ax.set_title(f'$ d_1 $ = {int(name_arr[1])}, $ d_2 $ = {int(name_arr[2])}, K = {int(name_arr[3])}', fontsize=20)
            ax.set_xlabel('Time', size=20)
            ax.set_ylabel('Regret', size=20)
            filename = f'Reg_vs_T_{int(name_arr[1])}_{int(name_arr[2])}_{int(name_arr[3])}_Avg_80k.png'
            plt.savefig(os.path.join(root, filename), dpi=200, format='png')
            plt.close()


def plot_num_arms_vs_regret_single_trial(folder, name, id):
    final_regret = {}
    idx = []
    _, ax = plt.subplots(1, 1, figsize=(6, 4))
    for root, dirs, files in os.walk(folder):
        for file in files:
            if (not file.startswith(name)) or (f'Trial_{id}' not in file):
                continue
            name_arr = file.split('_')
            if name_arr[3] == '5' and name_arr[4] == '5':
                df = pd.read_csv(os.path.join(root, file))
                print(id, os.path.join(root, file))
                if name_arr[5] in ['10', '25', '50', '100', '200', '300', '400']:
                    ls = df.cumsum().iloc[30000 - 1]
                else:
                    ls = df.cumsum().iloc[-1]
                idx.append(int(name_arr[5]))
                for k in ls.keys():
                    if k not in final_regret.keys():
                        final_regret[k] = [ls[k]]
                    else:
                        final_regret[k].append(ls[k])
        idx = np.array(idx)
        idx_sorted = np.argsort(idx)
        for k in final_regret.keys():
            ax.plot(idx[idx_sorted], np.array(final_regret[k])[idx_sorted], marker='o', markersize=8, linestyle='dashed', label=k, color=get_color(k))
        ax.grid()
        ax.legend()
        ax.set_title(f'Total Regret vs #Arms for Trial {id}')
        ax.set_xlabel('Number of Arms')
        ax.set_ylabel('Total Regret')
        plt.savefig(os.path.join(folder, f'Reg_vs_L_bar_plot_Trial_{id}.png'), dpi=200, format='png')
        plt.close()


def plot_num_arms_vs_regret_combined(folder, name):
    final_regret = {}
    idx = []
    _, ax = plt.subplots(1, 1, figsize=(6, 4))
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not file.startswith(name):
                continue
            name_arr = file.split('_')
            if name_arr[1] == '5' and name_arr[2] == '5':
                df = pd.read_csv(os.path.join(root, file))
                if name_arr[3] in ['10', '25', '50', '100', '200', '300', '400']:
                    ls = df.cumsum().iloc[30000 - 1]
                else:
                    ls = df.cumsum().iloc[-1]
                idx.append(int(name_arr[3]))
                for k in ls.keys():
                    if k not in final_regret.keys():
                        final_regret[k] = [ls[k]]
                    else:
                        final_regret[k].append(ls[k])
        idx = np.array(idx)
        idx_sorted = np.argsort(idx)
        for k in final_regret.keys():
            ax.plot(idx[idx_sorted], np.array(final_regret[k])[idx_sorted], marker='o', markersize=8, linestyle='dashed', label=k, color=get_color(k))
        ax.grid()
        ax.legend()
        ax.set_title('Total Regret vs #Arms')
        ax.set_xlabel('Number of Arms')
        ax.set_ylabel('Total Regret')
        plt.savefig(os.path.join(folder, 'Reg_vs_L_bar_plot.png'), dpi=200, format='png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='Testbench', help='Name of Experiment')
    parser.add_argument('--output', '-o', type=str, default='./Results', help='Output folder to store')
    parser.add_argument('--trials', '-t', type=int, default=5, help='Number of trial files')
    args = parser.parse_args()
    plot_num_arms_vs_regret_combined(args.output, args.name)
    plot_time_vs_regret_avg(args.output, args.name)
    for i in range(args.trials):
        plot_num_arms_vs_regret_single_trial(args.output, args.name, i+1)
        plot_time_vs_regret(args.output, args.name, i+1)
