import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_color

all_regret = {'LinUCB': np.zeros((10000000,)), 'DisLinUCB': np.zeros((10000000,)), 'HyRan': np.zeros((10000000,))}
time_steps = np.arange(1, 10000000+1)

i = 1
for root, _, files in os.walk('./New_Result'):
    for filename in files:
        if '10000000' in filename and '.csv' in filename:
            _, ax = plt.subplots(1, 1, figsize=(10, 6))
            regret_dict = pd.read_csv(os.path.join(root, filename))
            hylin = np.array(regret_dict['HyLinUCB'].cumsum())
            for k in regret_dict.columns:
                if k != 'HyLinUCB':
                    val = np.array(regret_dict[k].cumsum()) - hylin
                    all_regret[k] += val
                    ax.plot(time_steps, val, color=get_color(k), label=k)

            ax.legend()
            ax.grid()
            ax.set_title('Yahoo! Front Page', fontsize=16)
            ax.set_xlabel('Time', size=16)
            ax.set_ylabel('Relative Regret', size=16)
            ax.ticklabel_format(axis='both', scilimits=[0, 0])
            plt.savefig(os.path.join('./All_Sim_Final', f'Yahoo-Semi-Synthetic-10000000-{i}-Final.png'), dpi=200)
            i += 1

for k in all_regret.keys():
    all_regret[k] /= 5.0

_, ax = plt.subplots(1, 1, figsize=(10, 6))

for k in all_regret.keys():
    ax.plot(time_steps, all_regret[k], color=get_color(k), label=k)

ax.legend(fontsize=15)
ax.grid()
ax.set_title('Yahoo! Front Page', fontsize=18)
ax.set_xlabel('Time', size=18)
ax.set_ylabel('Relative Regret', size=18)
ax.ticklabel_format(axis='both', scilimits=[0, 0])
plt.savefig(os.path.join('./All_Sim_Final', f'Yahoo-Semi-Synthetic-10000000-Final.png'), dpi=200)
