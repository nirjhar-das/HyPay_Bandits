import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from multiprocessing.pool import Pool
from algorithms.linear import HyLinUCB_Offline, LinUCB_Offline, DisLinUCB_Offline

def generate_context(d_1, d_2, K, seed=0):
    rng_new = np.random.default_rng(seed)
    while(True):
        arm_feat = [(rng_new.uniform(-1/np.sqrt(d_1), 1/np.sqrt(d_1), size=(d_1,)),\
                    rng_new.uniform(-1/np.sqrt(d_2), 1/np.sqrt(d_2), size=(d_2,))) for _ in range(K)]

        yield arm_feat

def simulate_bandit(T, d1, d2, K, theta, beta, idx):
    hylin = HyLinUCB_Offline(d1, d2, K, 0.01, 1.0, 1.0, 1.0, 1.0, 0.01, 0.1)
    lin = LinUCB_Offline(d1, d2, K, 0.01, 1.0, 1.0, 1.0, 1.0, 0.01, 0.1)
    dislin = DisLinUCB_Offline(d1, d2, K, 0.01, 1.0, 1.0, 1.0, 1.0, 0.01, 0.1)
    con_gen = generate_context(d1, d2, K, seed=idx)
    rng_n = np.random.default_rng(idx + 478959)
    xxT_dict = {'HyLinUCB': 0.1 * np.eye(d1), 'LinUCB': 0.1 * np.eye(d1)}
    zzT_dict = {'HyLinUCB': [0.1 * np.eye(d2) for _ in range(K)], 'LinUCB': [0.1 * np.eye(d2) for _ in range(K)], 'DisLinUCB': [0.1 * np.eye(d1 + d2) for _ in range(K)]}
    xzT_dict = {'HyLinUCB': [0.1 * np.zeros((d1, d2)) for _ in range(K)], 'LinUCB': [0.1 * np.zeros((d1, d2)) for _ in range(K)]}
    V_eig_dict = {'HyLinUCB': [0.1], 'LinUCB': [0.1]}
    W_eig_dict = {'HyLinUCB': [[0.1] for _ in range(K)], 'LinUCB': [[0.1] for _ in range(K)], 'DisLinUCB': [[0.1] for _ in range(K)]}
    B_sing_dict = {'HyLinUCB': [[0.0] for _ in range(K)], 'LinUCB': [[0.0] for _ in range(K)]}
    for t in tqdm(range(T), position=idx+1):
        arm_feats = next(con_gen)
        
        a1 = hylin.predict(arm_feats)
        x, z = arm_feats[a1]
        xxT_dict['HyLinUCB'] += np.outer(x, x)
        zzT_dict['HyLinUCB'][a1] += np.outer(z, z)
        xzT_dict['HyLinUCB'][a1] += np.outer(x, z)
        V_eig_dict['HyLinUCB'].append(np.linalg.eigvalsh(xxT_dict['HyLinUCB'])[0])
        W_eig_dict['HyLinUCB'][a1].append(np.linalg.eigvalsh(zzT_dict['HyLinUCB'][a1])[0])
        B_sing_dict['HyLinUCB'][a1].append(np.linalg.norm(xzT_dict['HyLinUCB'][a1]))
        r1 = np.dot(theta, x) + np.dot(beta[a1], z) + rng_n.normal(0.0, 0.01)
        hylin.update(arm_feats, r1)

        a2 = lin.predict(arm_feats)
        x, z = arm_feats[a2]
        xxT_dict['LinUCB'] += np.outer(x, x)
        zzT_dict['LinUCB'][a2] += np.outer(z, z)
        xzT_dict['LinUCB'][a2] += np.outer(x, z)
        V_eig_dict['LinUCB'].append(np.linalg.eigvalsh(xxT_dict['LinUCB'])[0])
        W_eig_dict['LinUCB'][a2].append(np.linalg.eigvalsh(zzT_dict['LinUCB'][a2])[0])
        B_sing_dict['LinUCB'][a2].append(np.linalg.norm(xzT_dict['LinUCB'][a2]))
        r2 = np.dot(theta, x) + np.dot(beta[a2], z) + rng_n.normal(0.0, 0.01)
        hylin.update(arm_feats, r2)

        a3 = dislin.predict(arm_feats)
        x, z = arm_feats[a3]
        x_tilde = np.concatenate((x, z))
        zzT_dict['DisLinUCB'][a3] += np.outer(x_tilde, x_tilde)
        W_eig_dict['DisLinUCB'][a3].append(np.linalg.eigvalsh(zzT_dict['HyLinUCB'][a3])[0])
        r3 = np.dot(theta, x) + np.dot(beta[a3], z) + rng_n.normal(0.0, 0.01)
        hylin.update(arm_feats, r3)



        
    return V_eig_dict, W_eig_dict, B_sing_dict


def multi_bandit_simulation(n_trials, T, d1, d2, K, theta, beta):
    args_arr = []
    for i in range(n_trials):
        args_arr.append((T, d1, d2, K, theta, beta, i))
    with Pool(processes=4) as p:
        result = p.starmap(simulate_bandit, args_arr)
    
    fig1 = plt.figure(layout='tight')
    fig2 = plt.figure(layout='tight')
    gs1 = fig1.add_gridspec(2, 2)
    gs2 = fig1.add_gridspec(2, 2)
    ax1 = [fig1.add_subplot(gs1[0, :]), fig1.add_subplot(gs1[1, 0]), fig1.add_subplot(gs1[1, 1])]
    ax2 = [fig2.add_subplot(gs2[0, :]), fig2.add_subplot(gs2[1, 0]), fig2.add_subplot(gs2[1, 1])]
    # fig1, ax1 = plt.subplots(2, 1, figsize=(12, 10))
    # fig2, ax2 = plt.subplots(2, 1, figsize=(12, 10))
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 5))

    # max_slope_1, min_slope_1 = 0.0, 1000.0
    # max_slope_2, min_slope_2 = 0.0, 1000.0
    # max_len_1_W = 0
    # max_slope_1_W, min_slope_1_W = 0.0, 1000.0
    # max_slope_2_W, min_slope_2_W = 0.0, 1000.0
    # max_len_2_W = 0

    # max_slope_3, min_slope_3 = 0.0, 1000.0
    # max_len_3_W = 0

    x_ax_V = np.arange(1, T+2)

    for i in range(n_trials):
        V_eig_dict, W_eig_dict, B_sing_dict = result[i]

        ax1[0].plot(x_ax_V, V_eig_dict['HyLinUCB'], alpha=0.3, color='red')
        # max_slope_1 = max(max_slope_1, np.max(V_eig_dict['HyLinUCB'] / x_ax_V))
        # min_slope_1 = min(min_slope_1, np.min(V_eig_dict['HyLinUCB'] / x_ax_V))
        for j in range(K):
            ax1[1].plot(np.arange(1, len(W_eig_dict['HyLinUCB'][j]) + 1), W_eig_dict['HyLinUCB'][j], alpha=0.3, color='blue')
            # max_slope_1_W = max(max_slope_1_W, np.max(W_eig_dict['HyLinUCB'][j] / np.arange(1, len(W_eig_dict['HyLinUCB'][j]) + 1)))
            # min_slope_1_W = min(min_slope_1_W, np.min(W_eig_dict['HyLinUCB'][j] / np.arange(1, len(W_eig_dict['HyLinUCB'][j]) + 1)))
            # max_len_1_W = max(max_len_1_W, len(W_eig_dict['HyLinUCB'][j]))

            ax1[2].plot(np.arange(1, len(B_sing_dict['HyLinUCB'][j]) + 1), B_sing_dict['HyLinUCB'][j], alpha=0.3, color='blue')
        
        ax2[0].plot(x_ax_V, V_eig_dict['LinUCB'], alpha=0.3, color='red')
        # max_slope_2 = max(max_slope_1, np.max(V_eig_dict['LinUCB'] / x_ax_V))
        # min_slope_2 = min(min_slope_1, np.min(V_eig_dict['LinUCB'] / x_ax_V))
        for j in range(K):
            ax2[1].plot(np.arange(1, len(W_eig_dict['LinUCB'][j]) + 1), W_eig_dict['LinUCB'][j], alpha=0.3, color='blue')
            # max_slope_2_W = max(max_slope_2_W, np.max(W_eig_dict['LinUCB'][j] / np.arange(1, len(W_eig_dict['LinUCB'][j]) + 1)))
            # min_slope_2_W = min(min_slope_2_W, np.min(W_eig_dict['LinUCB'][j] / np.arange(1, len(W_eig_dict['LinUCB'][j]) + 1)))
            # max_len_2_W = max(max_len_2_W, len(W_eig_dict['LinUCB'][j]))

            ax2[2].plot(np.arange(1, len(B_sing_dict['LinUCB'][j]) + 1), B_sing_dict['LinUCB'][j], alpha=0.3, color='blue')

        for j in range(K):
            ax3.plot(np.arange(1, len(W_eig_dict['DisLinUCB'][j]) + 1), W_eig_dict['DisLinUCB'][j], alpha=0.3, color='blue')
            # max_slope_3 = max(max_slope_1, np.max(W_eig_dict['DisLinUCB'][j] / np.arange(1, len(W_eig_dict['DisLinUCB'][j]) + 1)))
            # min_slope_3 = min(max_slope_1, np.min(W_eig_dict['DisLinUCB'][j] / np.arange(1, len(W_eig_dict['DisLinUCB'][j]) + 1)))
            # max_len_3_W = max(max_len_3_W, len(W_eig_dict['DisLinUCB'][j]))


    # ax1[0].fill_between(x_ax_V, max_slope_1 * x_ax_V, min_slope_1 * x_ax_V, alpha = 0.1, color='red')
    # ax1[1].fill_between(np.arange(1, max_len_1_W + 1), max_slope_1_W * np.arange(1, max_len_1_W + 1), min_slope_1_W * np.arange(1, max_len_1_W + 1), alpha = 0.1, color='blue')
    # ax2[0].fill_between(x_ax_V, max_slope_2 * x_ax_V, min_slope_2 * x_ax_V, alpha = 0.1, color='red')
    # ax2[1].fill_between(np.arange(1, max_len_2_W + 1), max_slope_2_W * np.arange(1, max_len_2_W + 1), min_slope_2_W * np.arange(1, max_len_2_W + 1), alpha = 0.1, color='blue')
    # ax3.fill_between(np.arange(1, max_len_3_W + 1), max_slope_3 * np.arange(1, max_len_3_W + 1), min_slope_3 * np.arange(1, max_len_3_W + 1), alpha = 0.1, color='blue')

    ax1[0].set_xlabel('Time')
    ax1[0].set_ylabel('$ \lambda_{min} (V_t) $')
    ax1[0].grid()
    ax1[1].set_xlabel('Time')
    ax1[1].set_ylabel('$ \lambda_{min} (W_{i,t}) $')
    ax1[1].grid()
    ax1[2].set_xlabel('Time')
    ax1[2].set_ylabel('$ \sigma_{max} (B_{i,t}) $')
    ax1[2].grid()
    fig1.suptitle('HyLinUCB')


    ax2[0].set_xlabel('Time')
    ax2[0].set_ylabel('$ \lambda_{min} (V_t) $')
    ax2[0].grid()
    ax2[1].set_xlabel('Time')
    ax2[1].set_ylabel('$ \lambda_{min} (W_{i,t}) $')
    ax2[1].grid()
    ax2[2].set_xlabel('Time')
    ax2[2].set_ylabel('$ \sigma_{max} (B_{i,t}) $')
    ax2[2].grid()
    fig2.suptitle('LinUCB')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('$ \lambda_{min} (W_{i,t}) $')
    ax3.grid()
    fig3.suptitle('DisLinUCB')

    plt.draw()
    fig1.savefig(f'./Results/HyLinUCB_EigVal_{T}.png', dpi=200)
    fig2.savefig(f'./Results/LinUCB_EigVal_{T}.png', dpi=200)
    fig3.savefig(f'./Results/DisLinUCB_EigVal_{T}.png', dpi=200)



if __name__ == '__main__':
    rng = np.random.default_rng(46568961)
    d1, d2, K = 10, 10, 25
    T = 5000
    n_trials = 100
    theta = rng.uniform(-1.0 / np.sqrt(d1), 1.0 / np.sqrt(d1), size=((d1,)))
    beta = [rng.uniform(-1.0 / np.sqrt(d2), 1.0 / np.sqrt(d2), size=((d2,))) for _ in range(K)]

    multi_bandit_simulation(n_trials, T, d1, d2, K, theta, beta)