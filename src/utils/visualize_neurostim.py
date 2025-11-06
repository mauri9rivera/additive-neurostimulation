import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from collections import Counter
from itertools import combinations
from matplotlib import cm
from matplotlib.colors import Normalize


def load_results(filepath):

    data = np.load(filepath, allow_pickle=True)
    return data

def plot_kappas(npz_files, dataset, show=True):
    """
    Plot exploration (PP) and exploitation (PP_t) curves for each model/file in npz_files.
    - npz_files: list of filepaths (expected order: ExactGP, AdditiveGP, SobolGP, MHGP).
    - model_names: optional list of labels for legend (defaults to file basenames).
    - show: whether to plt.show() at the end (useful=interactive). Always saves.

    Each npz is expected to contain at least:
      - PP      : ndarray (e.g. (nSubjects, nEmgs, n_kappas, nRep, MaxQueries)) or similar
      - PP_t    : ndarray same shape as PP
      - kappas or kappa_vals or kappas_vals : 1D array of kappa values (length n_kappas)

    The function will average PP and PP_t across all axes except the kappa axis and the time axis
    (time axis is assumed to be the last axis).E
    """

    # colors for models in order: ExactGP, AdditiveGP, SobolGP, MHGP
    default_colors = ['red', 'blue', 'green']
    model_names = ['ExactGP', 'AdditiveGP', 'SobolGP']
    cmap_list = [cm.tab10, cm.tab10, cm.tab10, cm.tab10]
    emg_map = {
        'nhp': [6, 8, 4, 4],
        'rat': [6, 7, 8, 6, 5, 8],
        '5d_rat': [1, 1, 1, 1]
    }

    n = np.sum(emg_map[dataset])

    # Prepare save dir
    save_dir=f'output/neurostim_experiments/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    out_fname = f"{dataset}_kappa_comparison.svg"
    out_path = os.path.join(save_dir, out_fname)


    n_models = len(npz_files)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5), squeeze=False)
    axes = axes.flatten()

    for file_idx, filepath in enumerate(npz_files):
        ax = axes[file_idx]

        cmap = cmap_list[file_idx]
        
        data = np.load(filepath, allow_pickle=True)
        PP = data['PP']
        PP_t = data['PP_t']
        kappas =data['kappas']

        n_iters = PP.shape[-1]

        exploitations = np.zeros((len(kappas), n, n_iters))
        explorations = np.zeros((len(kappas), n, n_iters))
        
        # Convert torch tensors to numpy if necessary (some saved torch.cpu().numpy() already)
        PP = np.asarray(PP)
        PP_t = np.asarray(PP_t)

        n_kappas = int(kappas.shape[0])

        for k_i in range(n_kappas):
            jj=0
            for s_i in range(len(emg_map[dataset])):

                for muscle in range(emg_map[dataset][s_i]):

                    explorations[k_i, jj, :] = np.mean(PP[s_i, muscle, k_i], axis=0)
                    exploitations[k_i, jj, :] = np.mean(PP_t[s_i, muscle, k_i], axis=0)

                    jj+=1

        explorations = np.mean(explorations, axis=1)
        exploitations = np.mean(exploitations, axis=1)
        iterations = np.arange(1, n_iters+1)
        
        for k_idx, k_val in enumerate(kappas):

            c = cmap(k_idx)
            label = f"κ={float(k_val)} exploration"
            ax.plot(iterations, explorations[k_idx], linestyle='-', color=c, label=label)
            ax.plot(iterations, exploitations[k_idx], linestyle='--', color=c)

        ax.set_title(model_names[file_idx])
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Metric ')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True)
        ax.legend(fontsize='small', loc='lower right', ncol=1)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    plt.close(fig)

def optimization_metrics(data_exactgp, data_additivegp, data_sobolgp, kappas, dataset='nhp'):
    """
    Plot mean PP_t and PP across subjects for two datafiles.
    
    Exploration (PP_t) will be filled, exploitation (PP) will be dotted.
    
    Parameters:
    - data1: first dataset dict (contains 'PP_t' and 'PP' arrays)
    - data2: second dataset dict (contains 'PP_t' and 'PP' arrays)
    - kappa: exploration-exploitation tradeoff parameter
    - k_i: index for k_i dimension
    """
    datafiles = [data_exactgp, data_additivegp, data_sobolgp]
    colors = ['red', 'blue', 'green']
    labels = ['ExactGP', 'AdditiveGP', 'SobolGP']
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # ---------- Figure 1: Exploration & Exploitation ----------
    for idx, data in enumerate(datafiles):
        PP_t = data['PP_t']  # Shape: (n_subjects, ..., iterations, ...)
        PP = data['PP']

        n_subjects = PP_t.shape[0]
        # Compute mean and std across subjects and repeats

        mean_PP_t = PP_t[:, 0, kappas[idx], :, :].mean(axis=(0, 1))
        std_PP_t = PP_t[:, 0, kappas[idx], :, :].std(axis=(0, 1))
        
        mean_PP = PP[:, 0, kappas[idx], :, :].mean(axis=(0, 1))
        std_PP = PP[:, 0, kappas[idx], :, :].std(axis=(0, 1))
        
        # 95% confidence interval
        n_samples = n_subjects * PP_t.shape[3]
        conf_interval_PP_t = 1.96 * std_PP_t / np.sqrt(n_samples)
        conf_interval_PP = 1.96 * std_PP / np.sqrt(n_samples)
        
        x_values = np.arange(mean_PP_t.shape[0])
        
        # Plot exploration as dotted line
        ax.plot(x_values, mean_PP, color=colors[idx], linestyle='-', label=f'Exploration ({labels[idx]})')
        ax.fill_between(x_values, mean_PP - conf_interval_PP, mean_PP + conf_interval_PP, color=colors[idx], alpha=0.15)
        
        # Plot exploitation as filled area
        ax.plot(x_values, mean_PP_t, color=colors[idx], linestyle='--', label=f'Exploitation ({labels[idx]})')
        #ax.fill_between(x_values, mean_PP_t - conf_interval_PP_t, mean_PP_t + conf_interval_PP_t, color=colors[idx], alpha=0.3)
    
    ax.set_title(f"Exploration vs. Exploitation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Scores")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    
    output_dir = os.path.join('output', 'neurostim_experiments', dataset)
    os.makedirs(output_dir, exist_ok=True)

    perf_filename = f'explr-expl-{dataset}.svg'
    perf_path = os.path.join(output_dir, perf_filename)
    plt.savefig(perf_path, format='svg')
    plt.close(fig)


    # ---------- Figure 2: Regrets ----------
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    
    for idx, data in enumerate(datafiles):

        regret = data['REGRETS']  # Shape: (n_subjects, ..., iterations, ...)

        mean_regret = regret[:, 0, kappas[idx], :, :].mean(axis=(0, 1))
        std_regret = regret[:, 0, kappas[idx], :, :].std(axis=(0, 1))
        
        
        # 95% confidence interval
        n_samples = n_subjects * regret.shape[3]
        conf_interval_regret = 1.96 * std_regret / np.sqrt(n_samples)
        
        x_values = np.arange(mean_regret.shape[0])

        # Plot exploration as dotted line
        ax2.plot(x_values, mean_regret, color=colors[idx], linestyle='-', label=f'Regret ({labels[idx]})')
        ax2.fill_between(x_values, mean_regret - conf_interval_regret, mean_regret + conf_interval_regret, color=colors[idx], alpha=0.3)
        
    ax2.set_title("Log-Regret across runs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Regret")
    #ax2.set_yscale("log")
    ax.set_ylim(-10, 10)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax2.legend(loc='upper right', fontsize='small')

    regrets_filename = f'regrets-{dataset}.svg'
    regrets_path = os.path.join(output_dir, regrets_filename)
    plt.savefig(regrets_path, format='svg')
    plt.close(fig2)

def partition_metrics(data, dataset_type='nhp', threshold=0.05):

    if dataset_type == '5d_rat':
        d = 5
    else:
        d = 2
    # TODO: Get true Sobol second-order matrix
    true_sobols = False

    partitions = data['PARTITIONS']
    sobols = data['SOBOLS']

    # 3) Build G_true adjacency matrix using threshold 0.05 (upper diagonal)
    non_additive = true_sobols > threshold  
    additive = ~non_additive.copy()
    np.fill_diagonal(non_additive, False)
    np.fill_diagonal(additive, False)

    # Count unique undirected true edges (i<j)
    triu_idx = np.triu_indices(d, k=1)
    true_non_additive_count = int(np.sum(non_additive[triu_idx])) - f_obj.d
    true_additive_count = int(((d * (d - 1)) // 2) - true_non_additive_count)

        # Helper: build group_of array from partition P
    def partition_to_group_of(P):
        """
        Convert partition list-of-lists to group_of array of length d.
        If some indices missing, they remain -1 (shouldn't happen).
        """
        group_of = np.full(d, -1, dtype=int)
        for gid, grp in enumerate(P):
            for idx in grp:
                group_of[int(idx)] = gid
        return group_of

   # Helper: build experiment adjacency matrix from partition snapshot P
    def build_nonadditivity_graph(P):
        """
        P: list of lists (groups). Returns symmetric boolean adjacency matrix (d,d):
           adj[i,j] = True iff i and j are not in the same group (predicted non-additive / interaction).
        """
        adj = np.zeros((d, d), dtype=bool)
        group_of = partition_to_group_of(P)
        for i in range(d):
            for j in range(i + 1, d):
                adj[i, j] = (group_of[i] != group_of[j])
                # adj[j,i] will be mirrored below
        adj = adj | adj.T
        np.fill_diagonal(adj, False)
        return adj
    
    # 4) For each partition snapshot compute CC and CS
    cc_list = []
    cs_list = []

    prev_group_of = None
    partition_changed_flags = []

    for t, P in enumerate(partitions_all):

        iter_num = n_init + t + 1

        # compute if partition changed since last iteration
        current_group_of = partition_to_group_of(P)
        if prev_group_of is None:
            changed = True  # mark first snapshot as a change (so it can get markers if desired)
        else:
            changed = not np.array_equal(prev_group_of, current_group_of)
        partition_changed_flags.append(bool(changed))
        prev_group_of = current_group_of.copy()

        # predicted adjacency
        G_exp = build_nonadditivity_graph(P)
        non_additive_overlap = np.logical_and(non_additive, G_exp)
        non_additive_overlap_count = int(np.sum(non_additive_overlap[triu_idx]))

        # CC = fraction of true non-additive edges recovered
        if true_non_additive_count > 0:
            CC = non_additive_overlap_count / true_non_additive_count
        else:
            CC = 0.0

        # Count true additive (non-edge) that are predicted non-edge
        # predicted non-edge matrix:
        nonedge_pred = ~G_exp
        # true additive & predicted non-edge
        additive_overlap = np.logical_and(additive, nonedge_pred)
        np.fill_diagonal(additive_overlap, False)
        additive_overlap_count = int(np.sum(additive_overlap[triu_idx]))

        if true_additive_count > 0:
            CS = additive_overlap_count / true_additive_count
        else:
            CS = 0.0

        cc_list.append(CC)
        cs_list.append(CS)


    # 5) Plot CC and CS vs iteration
    plt.figure(figsize=(9, 5))
    update_iters = list(range(n_iter))
    x = np.array(update_iters)

    # n_init_padding
    cc_list = [0.0] * n_init + cc_list
    cs_list = [0.0] * n_init + cs_list
    partition_changed_flags = [False] * n_init + partition_changed_flags

    color = 'green' if model_cls.__name__ == 'SobolGP' else 'orange'

    # Plot CC (match of true edges)
    plt.plot(x, cc_list, label='CC (true-edge overlap)', linestyle='-', color=color)
    change_x = x[np.array(partition_changed_flags, dtype=bool)]
    change_cc = np.array(cc_list)[np.array(partition_changed_flags, dtype=bool)]
    if change_x.size > 0:
        plt.plot(change_x, change_cc, linestyle='None', marker='o', color=color)

    # Plot CS (match of true non-edges)
    plt.plot(x, cs_list, label='CS (true-nonedge overlap)', linestyle='--', color='dimgrey')
    change_cs = np.array(cs_list)[np.array(partition_changed_flags, dtype=bool)]
    if change_x.size > 0:
        plt.plot(change_x, change_cs, linestyle='None', marker='s', color='dimgrey')

    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction score')
    plt.ylim(0.0, 1.1)
    plt.title(f'Partition reconstruction for {dataset_type} kappa{kappa} |  (Additivity threshold={threshold})')
    plt.grid(True)
    plt.legend()
    
    output_dir = os.path.join('output', 'neurostim_experiments', dataset_type)
    os.makedirs(output_dir, exist_ok=True)
    fname = f'partition_recon_{dataset_type}_kappa{kappa}.png'
    outpath = os.path.join(output_dir, fname)
    plt.savefig(outpath, dpi=200)

   

    ### Figure 2: Sobol interaction traces
    pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    n_pairs = len(pairs)

    nrows = int(n_pairs)

    # Prepare sobol_est_list as list of (d,d) numpy arrays (one per iteration up to n_iter).
    sobol_est_list = []
    # Build list up to n_iter entries (pad with NaN matrices if missing)
    for t in range(n_iter):
        if t < len(sobol_interactions_all):
            candidate = sobol_interactions_all[t]
            if candidate.ndim == 0:
                candidate = np.ones((d, d), dtype=float)
            sobol_est_list.append(candidate)

    fig2, axes = plt.subplots(nrows, 1, figsize=(8, 1.8*nrows), squeeze=False)
    axes_flat = axes.flatten()
    color = 'green' if model_cls.__name__ == 'SobolGP' else 'orange'

    x_full = np.arange(1, n_iter + 1)
    for idx, (i, j) in enumerate(pairs):
        ax = axes_flat[idx]

        # Plot true as horizontal black line
        true_val = float(sobols_sym[i,j])
        ax.plot(x_full, [true_val]*x_full.shape[0], color='black', linestyle='--', linewidth=1.2)

        # Estimated sobol interaction curve
        est_vals = [1.0]*n_init
        for t in range(n_iter-n_init):
            est_mat = sobol_est_list[t]
            est_vals.append(float(est_mat[j][i]))
        est_vals = np.asarray(est_vals, dtype=float)
        ax.plot(x_full, est_vals, color=color, linestyle='-')

        ax.set_ylim(-0.1, 0.4)
        ax.set_xlim(1, n_iter)
        ax.set_title(f'x{i}-x{j} interaction')
        ax.grid(True, linestyle='--', linewidth=0.4)

    # wrap long suptitle into multiple lines so it doesn't overflow
    raw_title = f'Sobol reconstruction for {f_obj.d}-{f_obj.name} (kappa={kappa}) | {model_cls.__name__} Additivity threshold={threshold}'
    wrapped_title = textwrap.fill(raw_title, width=95)

    fig2.suptitle(wrapped_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    sobol_fname = f'sobol_recon_{model_cls.__name__}_{f_obj.d}d={f_obj.name}_kappa{kappa}.svg'
    outpath = os.path.join(output_dir, sobol_fname)
    fig2.savefig(outpath)
    plt.close(fig2)

    pass



if __name__ == '__main__': 

    files_rat = ['./output/neurostim_experiments/rat/rat_ExactGP_budget32_20reps.npz', #'./output/neurostim_experiments/rat/rat_ExactGP_budget32_20reps.npz',
             './output/neurostim_experiments/rat/rat_AdditiveGP_budget32_20reps.npz',
             './output/neurostim_experiments/rat/rat_neuralSobolGP_budget32_20reps.npz',
    ]

    files_nhp = ['./output/neurostim_experiments/nhp/nhp_ExactGP_budget96_20reps.npz',
             './output/neurostim_experiments/nhp/nhp_AdditiveGP_budget96_20reps.npz',
             './output/neurostim_experiments/nhp/nhp_neuralSobolGP_budget96_20reps.npz',
    ]


    files_5d_rat = ['./output/neurostim_experiments/5d_rat/5d_rat_ExactGP_budget100_20reps.npz',
            './output/neurostim_experiments/5d_rat/5d_rat_AdditiveGP_budget100_20reps.npz',
            './output/neurostim_experiments/5d_rat/5d_rat_neuralSobolGP_budget100_20reps.npz',
    ]

    d_simple = load_results(files_rat[0])
    d_additive = load_results(files_rat[1])
    d_sobol = load_results(files_rat[2])

    #optimization_metrics(d_simple, d_additive, d_sobol, [4, -1, -2], dataset='rat')

    plot_kappas(files_rat, 'rat')