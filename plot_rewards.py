import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from environment import HybridBandits

def create_plot(env:HybridBandits, output_folder, nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    x, y = 0, 0
    base = np.arange(1, env.L + 1)
    theta = env.parameters['theta']
    beta_arr = env.parameters['beta']
    for i in range(env.num_context):
        x = x % nrows
        y = y % ncols
        reward_arr = []
        for j, (a,b) in enumerate(env.arms[i]):
            reward_arr.append(np.dot(theta, a) + np.dot(beta_arr[j], b))
        ax[x][y].bar(base, reward_arr)
        ax[x][y].grid()
        x += 1
        y += 1
    plt.savefig(os.path.join(output_folder ,'Reward_Distribution.png'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to environment')
    parser.add_argument('-o', '--output', type=str, help='output folder')
    parser.add_argument('-r', '--nrows', type=int, help='number of rows')
    parser.add_argument('-c', '--ncols', type=int, help='number of cols')
    args = parser.parse_args()
    env = HybridBandits(load=args.path)
    create_plot(env, args.output, args.nrows, args.ncols)
