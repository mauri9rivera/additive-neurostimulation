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
import textwrap


DEVICE = 'cpu'

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
    ax.set_ylim(global_min, 1.05)
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

def partition_metrics(data, dataset_type, subject_idx, kappa_idx=2, figsize_per_ax=(4, 3), add_threshold=0.05):

    #1) Get True Sobol Interactions
    true_sobols = {
        'spinal': 
        {
            0: {
                0: {'x0 & x1': 0.0003},
                1: {'x0 & x1': 0.0191},
                2: {'x0 & x1': 0.0258},
                3: {'x0 & x1': 0.0113},
                4: {'x0 & x1': 0.1261},
                5: {'x0 & x1': 0.0666},
                6: {'x0 & x1': 0.0519},
                7: {'x0 & x1': 0.0766},
                },
            1: {
                0: {'x0 & x1': 0.0322},
                1: {'x0 & x1': 0.0066},
                2: {'x0 & x1': 0.0083},
                3: {'x0 & x1': 0.0083},
                4: {'x0 & x1': 0.0093},
                5: {'x0 & x1': 0.0854},
                6: {'x0 & x1': 0.0213},
                7: {'x0 & x1': 0.0716},
                8: {'x0 & x1': 0.0445},
                9: {'x0 & x1': 0.0215},
                },
            2: {
                0: {'x0 & x1': 0.0153},
                1: {'x0 & x1': 0.0040},
                2: {'x0 & x1': 0.1699},
                3: {'x0 & x1': 0.0482},
                4: {'x0 & x1': 0.0191},
                5: {'x0 & x1': 0.0638},
                6: {'x0 & x1': 0.0527},
                7: {'x0 & x1': 0.1404},
                8: {'x0 & x1': 0.0616},
                9: {'x0 & x1': 0.0416},
                },
            3: {
                0: {'x0 & x1': 0.0237},
                1: {'x0 & x1': 0.1484},
                2: {'x0 & x1': 0.0190},
                3: {'x0 & x1': 0.0096},
                4: {'x0 & x1': 0.0176},
                5: {'x0 & x1': 0.1632},
                6: {'x0 & x1': 0.0154},
                7: {'x0 & x1': 0.1115},
                8: {'x0 & x1': 0.1523},
                9: {'x0 & x1': 0.3587},
                },
            4: {
                0: {'x0 & x1': 0.0318},
                1: {'x0 & x1': 0.0318},
                2: {'x0 & x1': 0.0406},
                3: {'x0 & x1': 0.0348},
                4: {'x0 & x1': 0.1023},
                5: {'x0 & x1': 0.0991},
                6: {'x0 & x1': 0.0988},
                7: {'x0 & x1': 0.0720},
                8: {'x0 & x1': 0.2367},
                9: {'x0 & x1': 0.0544},
                },
            5: {
                0: {'x0 & x1': 0.0149},
                1: {'x0 & x1': 0.0464},
                2: {'x0 & x1': 0.0594},
                3: {'x0 & x1': 0.0467},
                4: {'x0 & x1': 0.0121},
                5: {'x0 & x1': 0.1108},
                6: {'x0 & x1': 0.1261},
                7: {'x0 & x1': 0.0569},
                8: {'x0 & x1': 0.2837},
                9: {'x0 & x1': 0.4564},
                },
            6: {
                0: {'x0 & x1': 0.1517},
                1: {'x0 & x1': 0.1559},
                2: {'x0 & x1': 0.1511},
                3: {'x0 & x1': 0.0946},
                4: {'x0 & x1': 0.1339},
                5: {'x0 & x1': 0.3739},
                6: {'x0 & x1': 0.3639},
                7: {'x0 & x1': 0.3527},
                8: {'x0 & x1': 0.2618},
                9: {'x0 & x1': 0.3277},
                },
            7: {
                0: {'x0 & x1': 0.5659},
                1: {'x0 & x1': 0.1477},
                2: {'x0 & x1': 0.5832},
                3: {'x0 & x1': 0.2785},
                4: {'x0 & x1': 0.5520},
                5: {'x0 & x1': 0.0614},
                6: {'x0 & x1': 0.5637},
                7: {'x0 & x1': 0.2725},
                }, 
            8: {
                0: {'x0 & x1': 0.1997},
                1: {'x0 & x1': 0.1284},
                2: {'x0 & x1': 0.3647},
                3: {'x0 & x1': 0.0902},
                4: {'x0 & x1': 0.0940},
                5: {'x0 & x1': 0.0617},
                6: {'x0 & x1': 0.5722},
                7: {'x0 & x1': 0.1314},
                }, 
            9: {
                0: {'x0 & x1': 0.4323},
                1: {'x0 & x1': 0.1189},
                2: {'x0 & x1': 0.3180},
                3: {'x0 & x1': 0.0189},
                4: {'x0 & x1': 0.3395},
                5: {'x0 & x1': 0.0367},
                6: {'x0 & x1': 0.5079},
                7: {'x0 & x1': 0.1018},
                }, 
            10: {
                0: {'x0 & x1': 0.2705},
                1: {'x0 & x1': 0.1235},
                2: {'x0 & x1': 0.1538},
                3: {'x0 & x1': 0.0582},
                4: {'x0 & x1': 0.3191},
                5: {'x0 & x1': 0.0909},
                6: {'x0 & x1': 0.2671},
                7: {'x0 & x1': 0.0563},
                },    
        },
        'nhp': 
        {
            0: {
                0: {'x0 & x1': 0.1150},
                1: {'x0 & x1': 0.1104},
                2: {'x0 & x1': 0.0710},
                3: {'x0 & x1': 0.1601},
                4: {'x0 & x1': 0.1296},
                5: {'x0 & x1': 0.1403},
                },
            1: {
                0: {'x0 & x1': 0.1937},
                1: {'x0 & x1': 0.2148},
                2: {'x0 & x1': 0.2206},
                3: {'x0 & x1': 0.2056},
                4: {'x0 & x1': 0.2037},
                5: {'x0 & x1': 0.2077},
                6: {'x0 & x1': 0.2652},
                7: {'x0 & x1': 0.2266},
                },
            2: {
                0: {'x0 & x1': 0.1305},
                1: {'x0 & x1': 0.0737},
                2: {'x0 & x1': 0.1713},
                3: {'x0 & x1': 0.1693},
                },
            3: {
                0: {'x0 & x1': 0.5885},
                1: {'x0 & x1': 0.3683},
                2: {'x0 & x1': 0.5994},
                3: {'x0 & x1': 0.3156},
                },
        },

        'rat': 
        {
            0: {
                0: {'x0 & x1': 0.1100},
                1: {'x0 & x1': 0.0980},
                2: {'x0 & x1': 0.0302},
                3: {'x0 & x1': 0.1282},
                4: {'x0 & x1': 0.0759},
                5: {'x0 & x1': 0.1031},
                },
            1: {
                0: {'x0 & x1': 0.3427},
                1: {'x0 & x1': 0.0576},
                2: {'x0 & x1': 0.6300},
                3: {'x0 & x1': 0.1310},
                4: {'x0 & x1': 0.0768},
                5: {'x0 & x1': 0.0645},
                6: {'x0 & x1': 0.1054},
                },
            2: {
                0: {'x0 & x1': 0.1123},
                1: {'x0 & x1': 0.0183},
                2: {'x0 & x1': 0.0342},
                3: {'x0 & x1': 0.0296},
                4: {'x0 & x1': 0.1874},
                5: {'x0 & x1': 0.0311},
                6: {'x0 & x1': 0.0340},
                7: {'x0 & x1': 0.0621},
                },
            3: {
                0: {'x0 & x1': 0.1828},
                1: {'x0 & x1': 0.1934},
                2: {'x0 & x1': 0.0723},
                3: {'x0 & x1': 0.1000},
                4: {'x0 & x1': 0.1447},
                5: {'x0 & x1': 0.0689},
                },
            4: {
                0: {'x0 & x1': 0.3164},
                1: {'x0 & x1': 0.0631},
                2: {'x0 & x1': 0.2639},
                3: {'x0 & x1': 0.1344},
                4: {'x0 & x1': 0.2682},
                },
            5: {
                0: {'x0 & x1': 0.0552},
                1: {'x0 & x1': 0.0458},
                2: {'x0 & x1': 0.2341},
                3: {'x0 & x1': 0.3674},
                4: {'x0 & x1': 0.0883},
                5: {'x0 & x1': 0.1472},
                6: {'x0 & x1': 0.2491},
                6: {'x0 & x1': 0.4263},
                },
        },

        '5d_rat': 
        {
            0: {
                0: {'x0 & x1':  0.0004, 'x0 & x2':  0.0016, 'x0 & x3':  0.0248, 'x0 & x4':  0.0103, 
                    'x1 & x2':  0.0003, 'x1 & x3':  0.0038, 'x1 & x4':  0.0029, 'x2 & x3':  0.0060, 
                    'x2 & x4':   0.0014, 'x3 & x4':  0.0286},
                1: {'x0 & x1':  0.0002, 'x0 & x2': 0.0033, 'x0 & x3':  0.0506, 'x0 & x4':  0.0020, 
                    'x1 & x2':  0.0002, 'x1 & x3':  0.0025, 'x1 & x4':  0.0018, 'x2 & x3':  0.0133, 
                    'x2 & x4':  0.0020, 'x3 & x4':   0.0283},
                2: {'x0 & x1':  0.0000, 'x0 & x2':   0.0171, 'x0 & x3':  0.0784, 'x0 & x4':  0.0045, 
                    'x1 & x2':  0.0020, 'x1 & x3':  0.0026, 'x1 & x4':  0.0007, 'x2 & x3':  0.1029, 
                    'x2 & x4':  0.0080, 'x3 & x4':  0.0451},
                3: {'x0 & x1':  0.0002, 'x0 & x2':  0.0072, 'x0 & x3':  0.0437, 'x0 & x4':  0.0016, 
                    'x1 & x2':  0.0004, 'x1 & x3':  0.0005, 'x1 & x4':  0.0000, 'x2 & x3':  0.0254, 
                    'x2 & x4':  0.0018, 'x3 & x4':  0.0209},
                4: {'x0 & x1':  0.0008, 'x0 & x2':  0.0123, 'x0 & x3':  0.0620, 'x0 & x4':  0.0015, 
                    'x1 & x2':  0.0034, 'x1 & x3':  0.0001, 'x1 & x4':  0.0004, 'x2 & x3':  0.0391, 
                    'x2 & x4':  0.0022, 'x3 & x4':  0.0145},
                },
            1: {
                0: {'x0 & x1':  0.0136, 'x0 & x2':  0.0000, 'x0 & x3':  0.0401, 'x0 & x4':  0.1104, 
                    'x1 & x2':  0.0012, 'x1 & x3':  0.0233, 'x1 & x4':  0.0497, 'x2 & x3': 0.0001, 
                    'x2 & x4':  0.0004, 'x3 & x4':  0.1196},
                1: {'x0 & x1':  0.0070, 'x0 & x2':  0.0055, 'x0 & x3':  0.0018, 'x0 & x4':  0.0168, 
                    'x1 & x2':  0.0087, 'x1 & x3':  0.0113, 'x1 & x4':  0.0035, 'x2 & x3':  0.0089, 
                    'x2 & x4':  0.0092, 'x3 & x4':  0.0346},
                2: {'x0 & x1':  0.0502, 'x0 & x2':  0.0032, 'x0 & x3': 0.0025, 'x0 & x4':  0.0073, 
                    'x1 & x2':  0.0145, 'x1 & x3':  0.0183, 'x1 & x4':  0.0074, 'x2 & x3':  0.0132, 
                    'x2 & x4':  0.0299, 'x3 & x4':  0.0271},
                3: {'x0 & x1':  0.0128, 'x0 & x2':  0.0016, 'x0 & x3':  0.0205, 'x0 & x4':  0.1308, 
                    'x1 & x2':  0.0012, 'x1 & x3':  0.0107, 'x1 & x4': 0.0284, 'x2 & x3':   0.0000, 
                    'x2 & x4':  0.0064, 'x3 & x4':  0.0635},
                },
            2: {
                0: {'x0 & x1':  0.0974, 'x0 & x2':  0.0000, 'x0 & x3':  0.0006, 'x0 & x4': 0.0015, 
                    'x1 & x2':  0.0001, 'x1 & x3':  0.0017, 'x1 & x4':  0.0024, 'x2 & x3':  0.0001, 
                    'x2 & x4':  0.0001, 'x3 & x4':  0.0041},
                1: {'x0 & x1':  0.2198, 'x0 & x2':  0.0000, 'x0 & x3':  0.0022, 'x0 & x4':  0.0022, 
                    'x1 & x2':  0.0005, 'x1 & x3':  0.0041, 'x1 & x4':  0.0043, 'x2 & x3':  0.0002, 
                    'x2 & x4':  0.0003, 'x3 & x4':  0.0042},
                2: {'x0 & x1':  0.2773, 'x0 & x2':  0.0003, 'x0 & x3':  0.0016, 'x0 & x4':  0.0037, 
                    'x1 & x2':  0.0001, 'x1 & x3':  0.0015, 'x1 & x4': 0.0030, 'x2 & x3':  0.0001, 
                    'x2 & x4':  0.0001, 'x3 & x4':  0.0025},
                3: {'x0 & x1':  0.1629, 'x0 & x2':  0.0001, 'x0 & x3': 0.0014, 'x0 & x4':  0.0019, 
                    'x1 & x2':  0.0008, 'x1 & x3':  0.0031, 'x1 & x4':  0.0043, 'x2 & x3':  0.0002, 
                    'x2 & x4':  0.0002, 'x3 & x4':  0.0034},
                },
        }
    }
    d = 5 if dataset_type == '5d_rat' else 2
    pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    n_interactions = len(pairs)
    partitions = data['PARTITIONS']
    surrogate_sobols = data['SOBOLS']

    options = set_experiment(dataset_type)
    n_emgs = options['n_emgs'][subject_idx]

    fig_w = figsize_per_ax[0] * n_interactions
    fig_h = figsize_per_ax[1] * n_emgs

    fig, axes = plt.subplots(n_emgs, n_interactions, figsize=(fig_w + 3, fig_h + 3), squeeze=False )
    labels = ['True interaction', 'Additivity threshold', 'Predicted interaction']

    for idx, (i, j) in enumerate(pairs):

        for e_i in range(options['n_emgs'][subject_idx]):

            ax = axes[e_i, idx]            


            x_full = np.arange(1, surrogate_sobols.shape[-1] + 1)

            # Plot true sobol as horizontal black line
            true_sobol = true_sobols[dataset_type][subject_idx][e_i][f'x{i} & x{j}']
            true_label = ax.plot(x_full, [true_sobol]*x_full.shape[0], color='black', linestyle='--', linewidth=1.2, label='True interaction')
            threshold_label = ax.plot(x_full, [add_threshold]*x_full.shape[0], color='red', linestyle='--', linewidth=1.2, label='Additivity threshold')
            
            # Estimated sobol interaction curve
            surrogate_mean = []
            for rep_i in range(surrogate_sobols.shape[3]):

                sobol_trace = [1.0]
                for t in range(surrogate_sobols.shape[-1]-1):
                    surrogate_estimation = surrogate_sobols[subject_idx, e_i, kappa_idx, rep_i, t]
                    sobol_trace.append(float(surrogate_estimation[i][j]))

                sobol_trace = np.asarray(sobol_trace, dtype=float)
                surrogate_mean.append(sobol_trace)
                ax.plot(x_full, sobol_trace, color='green', linestyle='-', alpha=0.10)

            surrogate_mean = np.asarray(surrogate_mean)
            surrogate_mean = np.mean(surrogate_mean, axis=0)
            prediction_label = ax.plot(x_full, surrogate_mean, color='green', linestyle='-', label='Predicted interaction')

            ax.set_ylim(-0.1, 1.0)
            ax.set_xlim(1, surrogate_sobols.shape[-1])
            ax.set_title(f'x{i}-x{j} interaction')
            ax.grid(True, linestyle='--', linewidth=0.4)

    # wrap long suptitle into multiple lines so it doesn't overflow
    raw_title = f'Sobol reconstruction for {dataset_type} | SobolGP (kappa={data["kappas"][kappa_idx]}), Sobol threshold={add_threshold}'
    wrapped_title = textwrap.fill(raw_title, width=40)

    fig.suptitle(wrapped_title, fontsize=14)
    fig.legend([true_label, threshold_label, prediction_label], labels=labels, loc="upper right", bbox_to_anchor=(1, 0.95))
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_dir = os.path.join('output', 'neurostim_experiments', dataset_type)
    sobol_fname = f'sobol_reconstruction_{dataset_type}_subj{subject_idx}_kappa{data["kappas"][kappa_idx]}.svg'
    outpath = os.path.join(output_dir, sobol_fname)
    fig.savefig(outpath)
    plt.close(fig)

def set_experiment(dataset_type):

    options = {}
    if dataset_type == 'nhp':
        options['noise_min']= 0.009 #Non-zero to avoid numerical instability
        options['kappa']=7
        options['rho_high']=3.0
        options['rho_low']=0.1
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.011
        options['n_subjects']=4
        options['n_queries']=96
        options['n_emgs'] = [6, 8, 4, 4]
        options['n_dims'] = 2
    elif dataset_type == 'rat':
        options['noise_min']=0.05
        options['kappa']=3.8
        options['rho_high']=8
        options['rho_low']=0.001
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.055
        options['n_subjects']=6
        options['n_queries']=32
        options['n_emgs'] = [6, 7, 8, 6, 5, 8]
        options['n_dims'] = 2
    elif dataset_type == '5d_rat':
        options['noise_min']=0.05
        options['kappa']=3.8
        options['rho_high']=8
        options['rho_low']=0.001
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.055
        options['n_subjects']=3
        options['n_queries']= 100
        options['n_emgs'] = [5, 4, 4]
        options['n_dims'] = 5
    elif dataset_type == 'spinal':
        options['noise_min']=0.05
        options['kappa']=3.8
        options['rho_high']=8
        options['rho_low']=0.001
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.055
        options['n_subjects']=11
        options['n_queries']=64
        options['n_emgs'] = [8, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8]
        options['n_dims'] = 2
    
    options['device'] = DEVICE
    options['n_reps'] = 20
    options['n_rnd'] = 1

    return options



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

    #d_simple = load_results(files_spinal[0])
    #d_additive = load_results(files_spinal[1])
    d_sobol = load_results(files_spinal[2])

    #optimization_metrics(d_simple, d_additive, d_sobol, kappas=[-2, -2, -1], dataset='spinal')
    
    partition_metrics(d_sobol, 'spinal', 8, -1)
    #plot_kappas(files_spinal, 'spinal')