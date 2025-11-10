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

from src.neurostim import *

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
        '5d_rat': [1, 1, 1, 1],
        'spinal': [8, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8]
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
    
    fig, ax = plt.subplots(figsize=(10, 8))

    global_min = 1.0
    
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

        k = data['kappas'][kappas[idx]]
        
        # Plot exploration as dotted line
        ax.plot(x_values, mean_PP, color=colors[idx], linestyle='-', label=f'Exploration ({labels[idx]}) | kappa {k}')
        ax.fill_between(x_values, mean_PP - conf_interval_PP, mean_PP + conf_interval_PP, color=colors[idx], alpha=0.15)
        
        # Plot exploitation as filled area
        ax.plot(x_values, mean_PP_t, color=colors[idx], linestyle='--')
        #ax.fill_between(x_values, mean_PP_t - conf_interval_PP_t, mean_PP_t + conf_interval_PP_t, color=colors[idx], alpha=0.3)

        #update global min exploration
        if mean_PP[0] < global_min:
            global_min = mean_PP[0]
    
    ax.set_title(f"Exploration vs. Exploitation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Scores")
    ax.set_ylim(global_min, 1.1)
    ax.legend(loc="lower right")
    
    output_dir = os.path.join('output', 'neurostim_experiments', dataset)
    os.makedirs(output_dir, exist_ok=True)

    perf_filename = f'explr-expl-{dataset}.svg'
    perf_path = os.path.join(output_dir, perf_filename)
    plt.savefig(perf_path, format='svg')
    plt.close(fig)


    # ---------- Figure 2: Regrets ----------
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
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
        
    ax2.set_title("Regret across runs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel(" Log Regret")
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

    files_spinal = ['./output/neurostim_experiments/spinal/spinal_ExactGP_budget64_20reps_wbaseline.npz',
            './output/neurostim_experiments/spinal/spinal_AdditiveGP_budget64_20reps_wbaseline.npz',
            './output/neurostim_experiments/spinal/spinal_neuralSobolGP_budget64_20reps_wbaseline.npz',
            ]

    d_simple = load_results(files_5d_rat[0])
    d_additive = load_results(files_5d_rat[1])
    d_sobol = load_results(files_5d_rat[2])

    optimization_metrics(d_simple, d_additive, d_sobol, [-2, 3, -2], dataset='5d_rat')

    #plot_kappas(files_spinal, 'spinal')