import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from algorithms.linear import HyLinUCB_Offline, LinUCB_Offline

rng = np.random.default_rng(46568961)

def generate_context(d_1, d_2, K):
    x_arr = [rng.uniform(-1/np.sqrt(d_1), 1/np.sqrt(d_1), size=(d_1,)) for _ in range(K)]
    z_arr = [rng.uniform(-1/np.sqrt(d_2), 1/np.sqrt(d_2), size=(d_2,)) for _ in range(K)]

    return x_arr, z_arr

def simulate_bandit(T, n_samples, n_keep, d1, d2, K):
    hylin = [HyLinUCB_Offline(d1, d2, K, 0.01, 1.0, 1.0, 1.0, 1.0, 0.01, 0.1) for _ in range(n_keep)]
    lin = [LinUCB_Offline(d1, d2, K, 0.01, 1.0, 1.0, 1.0, 1.0, 0.01, 0.1) for _ in range(n_keep)]
    theta = rng.uniform(-1.0 / np.sqrt(d1), 1.0 / np.sqrt(d1), size=((d1,)))
    beta = [rng.uniform(-1.0 / np.sqrt(d2), 1.0 / np.sqrt(d2), size=((d2,))) for _ in range(K)]
    eig_x_1 = np.zeros((n_keep, T))
    eig_z_1 = np.zeros((n_keep, T))
    eig_x_2 = np.zeros((n_keep, T))
    eig_z_2 = np.zeros((n_keep, T))
    fro_xz_1 = np.zeros((n_keep, T))
    fro_xz_2 = np.zeros((n_keep, T))
    eig_V_1 = np.zeros((T,))
    eig_W_1 = [[] for _ in range(K)]
    sing_B_1 = [[] for _ in range(K)]
    eig_V_2 = np.zeros((T,))
    eig_W_2 = [[] for _ in range(K)]
    sing_B_2 = [[] for _ in range(K)]
    xxT_1_1 = 0.1 * np.eye(d1)
    xxT_2_1 = 0.1 * np.eye(d1)
    zzT_1_1 = [0.1 * np.eye(d2) for _ in range(K)]
    zzT_2_1 = [0.1 * np.eye(d2) for _ in range(K)]
    xzT_1_1 = [np.zeros((d1, d2)) for _ in range(K)]
    xzT_2_1 = [np.zeros((d1, d2)) for _ in range(K)]
    for t in tqdm(range(T)):
        keep_idx = rng.integers(n_samples, size=(n_keep,))
        xxT_1 = [np.zeros((d1, d1)) for _ in range(n_keep)]
        zzT_1 = [np.zeros((d2, d2)) for _ in range(n_keep)]
        xzT_1 = [np.zeros((d1, d2)) for _ in range(n_keep)]
        xxT_2 = [np.zeros((d1, d1)) for _ in range(n_keep)]
        zzT_2 = [np.zeros((d2, d2)) for _ in range(n_keep)]
        xzT_2 = [np.zeros((d1, d2)) for _ in range(n_keep)]
        arms_keep = [None for _ in range(n_keep)]
        r1 = [None for _ in range(n_keep)]
        r2 = [None for _ in range(n_keep)]
        for n in range(n_samples):
            x_arr, z_arr = generate_context(d1, d2, K)
            arm_feats = [(x, z) for x, z in zip(x_arr, z_arr)]
            for k in range(n_keep):
                a1 = lin[k].predict(arm_feats)
                a2 = hylin[k].predict(arm_feats)
                xxT_1[k] += np.outer(x_arr[a1], x_arr[a1])
                zzT_1[k] += np.outer(z_arr[a1], z_arr[a1])
                xzT_1[k] += np.outer(x_arr[a1], z_arr[a1])
                xxT_2[k] += np.outer(x_arr[a2], x_arr[a2])
                zzT_2[k] += np.outer(z_arr[a2], z_arr[a2])
                xzT_2[k] += np.outer(x_arr[a2], z_arr[a2])
                if n == keep_idx[k]:
                    r1[k] = np.dot(x_arr[a1], theta) + np.dot(z_arr[a1], beta[a1]) + rng.normal(0.0, 0.01)
                    r2[k] = np.dot(x_arr[a2], theta) + np.dot(z_arr[a2], beta[a2]) + rng.normal(0.0, 0.01)
                    arms_keep[k] = arm_feats
        
        dum_zzT_1 = [np.zeros((d2, d2)) for _ in range(K)]
        dum_zzT_2 = [np.zeros((d2, d2)) for _ in range(K)]
        dum_xzT_1 = [np.zeros((d1, d2)) for _ in range(K)]
        dum_xzT_2 = [np.zeros((d1, d2)) for _ in range(K)]
        count_1 = [0 for _ in range(K)]
        count_2 = [0 for _ in range(K)]
        for k in range(n_keep):
            eig_x_1[k, t] = np.linalg.eigvalsh(xxT_1[k] / n_samples)[0]
            eig_x_2[k, t] = np.linalg.eigvalsh(xxT_2[k] / n_samples)[0]
            eig_z_1[k, t] = np.linalg.eigvalsh(zzT_1[k] / n_samples)[0]
            eig_z_2[k, t] = np.linalg.eigvalsh(zzT_2[k] / n_samples)[0]
            fro_xz_1[k, t] = np.linalg.norm(xzT_1[k] / n_samples, ord='fro')
            fro_xz_2[k, t] = np.linalg.norm(xzT_2[k] / n_samples, ord='fro')
            a1, a2 = lin[k].predict(arms_keep[k]), hylin[k].predict(arms_keep[k])
            count_1[a1] += 1
            count_2[a2] += 1
            xxT_1_1 += (np.outer(arms_keep[k][a1][0], arms_keep[k][a1][0]) / n_keep)
            xxT_2_1 += (np.outer(arms_keep[k][a2][0], arms_keep[k][a2][0]) / n_keep)
            dum_zzT_1[a1] += np.outer(arms_keep[k][a1][1], arms_keep[k][a1][1])
            dum_zzT_2[a2] += np.outer(arms_keep[k][a2][1], arms_keep[k][a2][1])
            dum_xzT_1[a1] += np.outer(arms_keep[k][a1][0], arms_keep[k][a1][1])
            dum_xzT_2[a2] += np.outer(arms_keep[k][a2][0], arms_keep[k][a2][1])
            lin[k].update(arms_keep[k], r1[k])
            hylin[k].update(arms_keep[k], r2[k])
        
        eig_V_1[t] = np.linalg.eigvalsh(xxT_1_1)[0]
        eig_V_2[t] = np.linalg.eigvalsh(xxT_2_1)[0]
        for i in range(K):
            if count_1[i] != 0:
                zzT_1_1[i] += (dum_zzT_1[i] / count_1[i])
                eig_W_1[i].append(np.linalg.eigvalsh(zzT_1_1[i])[0])
                xzT_1_1[i] += (dum_xzT_1[i] / count_1[i])
                sing_B_1[i].append(np.linalg.norm(xzT_1_1[i], ord=2))
            if count_2[i] != 0:
                zzT_2_1[i] += (dum_zzT_2[i] / count_2[i])
                eig_W_2[i].append(np.linalg.eigvalsh(zzT_2_1[i])[0])
                xzT_2_1[i] += (dum_xzT_2[i] / count_2[i])
                sing_B_2[i].append(np.linalg.norm(xzT_2_1[i], ord=2))
    
    result_dict = {'HyLinUCB': {'per_t_eig_x': eig_x_2, 'per_t_eig_z': eig_z_2,\
                                'per_t_fro_xz': fro_xz_2, 'eig_V': eig_V_2, 'eig_W_arr': eig_W_2,\
                                'sing_B_arr': sing_B_2},\
                    'LinUCB': {'per_t_eig_x': eig_x_1, 'per_t_eig_z': eig_z_1,\
                                'per_t_fro_xz': fro_xz_1, 'eig_V': eig_V_1, 'eig_W_arr': eig_W_1,\
                                'sing_B_arr': sing_B_1}}
    
    return result_dict


def plot_result(result_dict, T, K, n_keep):
    x_arr = np.arange(1, T+1)
    viridis = cm.get_cmap('viridis', 3*K)
    colors = viridis(np.linspace(0.0, 1.0, 3*K))
    fig, ax = plt.subplots(3, 2, figsize=(25, 40))
    d1 = result_dict['HyLinUCB']
    d2 = result_dict['LinUCB']
    for i in range(n_keep):
        if i == 0:
            ax[0][0].plot(x_arr, d1['per_t_eig_x'][i], color='blue', marker='.', label='HyLinUCB')
            ax[0][0].plot(x_arr, d2['per_t_eig_x'][i], color='red', marker='.', label='LinUCB')
            ax[0][1].plot(x_arr, d1['per_t_eig_z'][i], color='blue', marker='.', label='HyLinUCB')
            ax[0][1].plot(x_arr, d2['per_t_eig_z'][i], color='red', marker='.', label='LinUCB')
            ax[1][0].plot(x_arr, d1['per_t_fro_xz'][i], color='blue', marker='.', label='HyLinUCB')
            ax[1][0].plot(x_arr, d2['per_t_fro_xz'][i], color='red', marker='.', label='LinUCB')            
        else:
            ax[0][0].plot(x_arr, d1['per_t_eig_x'][i], color='blue', marker='.')
            ax[0][0].plot(x_arr, d2['per_t_eig_x'][i], color='blue', marker='.')
            ax[0][1].plot(x_arr, d1['per_t_eig_z'][i], color='blue', marker='.')
            ax[0][1].plot(x_arr, d2['per_t_eig_z'][i], color='red', marker='.')
            ax[1][0].plot(x_arr, d1['per_t_fro_xz'][i], color='blue', marker='.')
            ax[1][0].plot(x_arr, d2['per_t_fro_xz'][i], color='red', marker='.')

    for j in range(K):
        ax[1][1].plot(x_arr, d1['eig_V'], color=colors[j], marker='.', label='HyLinUCB')
        ax[1][1].plot(x_arr, d2['eig_V'], color=colors[j + K], marker='.', label='LinUCB')
        ax[2][0].plot(np.arange(1, len(d1['eig_W_arr'][j]) + 1), d1['eig_W_arr'][j], color=colors[j], marker='.', label='HyLinUCB')
        ax[2][0].plot(np.arange(1, len(d2['eig_W_arr'][j]) + 1), d2['eig_W_arr'][j], color=colors[j + 2*K], marker='.', label='LinUCB')
        ax[2][1].plot(np.arange(1, len(d1['sing_B_arr'][j]) + 1), d1['sing_B_arr'][j], color=colors[j], marker='.', label='HyLinUCB')
        ax[2][1].plot(np.arange(1, len(d2['sing_B_arr'][j]) + 1), d2['sing_B_arr'][j], color=colors[j + 2*K], marker='.', label='LinUCB')

    ax[0][0].set_title('Min eigval of E[x_t x_t^T]')
    ax[0][0].set_ylabel('Value')
    ax[0][0].set_xlabel('Time')

    ax[0][1].set_title('Min eigval of E[z_t z_t^T]')
    ax[0][1].set_ylabel('Value')
    ax[0][1].set_xlabel('Time')

    ax[1][0].set_title('Frob norm of E[x_t z_t^T]')
    ax[1][0].set_ylabel('Value')
    ax[1][0].set_xlabel('Time')

    ax[1][1].set_title('Avg min eigval of V_t')
    ax[1][1].set_ylabel('Value')
    ax[1][1].set_xlabel('Time')

    ax[2][0].set_title('Avg min eigval of W_{i,t}')
    ax[2][0].set_ylabel('Value')
    ax[2][0].set_xlabel('Time')

    ax[2][1].set_title('Avg max sing. val. of B_{i,t}')
    ax[2][1].set_ylabel('Value')
    ax[2][1].set_xlabel('Time')

    for i in range(3):
        for j in range(2):
            ax[i][j].legend()
            ax[i][j].grid()
    
    plt.savefig(os.path.join('./New_Result/Eigenvalue_Simulation.png'), dpi=200)


if __name__ == '__main__':
    result = simulate_bandit(2500, 50, 20, 4, 2, 20)
    plot_result(result, 2500, 10, 20)