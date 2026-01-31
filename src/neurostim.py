import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import sys
import os
import torch
import gpytorch
import argparse
import time
import scipy.io
import pickle
import math
import gc

from SALib.analyze import sobol as salib_sobol
from SALib.sample import sobol as sobol_sampler

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

### MODULES HANDLING ###
from utils.neurostim_datasets import *
from models.gaussians import NeuralSobolGP, NeuralAdditiveGP, NeuralExactGP, optimize, maximize_acq
from models.sobols import NeuralSobol

warnings.filterwarnings("ignore", category=FutureWarning, module="SALib.util")

### Method to calculate true Sobol Interactions

def build_surrogate(X, Y, type, 
                    D=2, pce_degree=3):
    if type == "rf":
        model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
        model.fit(X, Y)

        def predict(Xq):
            model.predict(Xq)

        return predict
    
    elif type == "gp":

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(D), nu=2.5) + WhiteKernel(noise_level=1e-6)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=2)
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpr.fit(X, Y)
        
        def predict(Xq):
            # GPR predict returns mean, std; use mean
            mu = gpr.predict(Xq, return_std=False)
            return mu
        
        return predict
    
    elif type == 'pce':
        degree = pce_degree
        polyf = PolynomialFeatures(degree=degree, include_bias=True)
        Xpoly = polyf.fit_transform(X)
        ridge = Ridge(alpha=1e-6, fit_intercept=False)
        ridge.fit(Xpoly, Y)
        def predict(Xq):
            return ridge.predict(polyf.transform(Xq))
        return predict

def sobol_interactions(dataset_type, surrogate='rf', N=4096):

    print(f'Sobol 2nd order interactions for {dataset_type}')
    options = set_experiment(dataset_type)

    S2 = []

    for m_i in range(options['n_subjects']):

        subject = load_data(dataset_type, m_i)
        X = subject['ch2xy'].astype(float)

        s1 = []
        s2 = []
        n_reps = 30

        for e_i in range(len(subject['emgs'])):

            Y = subject['sorted_respMean'][:,e_i]
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            X_scaled = (X - X_min) / (X_max - X_min)

            D = X_scaled.shape[1]
            problem = {'num_vars': D, 'names': [f"x{i}" for i in range(D)], 'bounds': [[0,1]]*D}
            
            
            predictor = build_surrogate(X_scaled, Y, surrogate, D = D)
            params = sobol_sampler.sample(problem, N, calc_second_order=True)  
            Y_samp = predictor(params)

            Si = salib_sobol.analyze(problem, Y_samp, calc_second_order=True, print_to_console=False)  
            s1.append(Si['S1'])
            s2.append(Si['S2'])

            emg_avg = []
            for rep_i in range(n_reps):
                predictor = build_surrogate(X_scaled, Y, surrogate, D = D)
                params = sobol_sampler.sample(problem, N, calc_second_order=True)  
                Y_samp = predictor(params)
                Si = salib_sobol.analyze(problem, Y_samp, calc_second_order=True, print_to_console=False)  
                emg_avg.append(Si['S2'])

            for i in range(D):
                for j in range(i+1, D):

                    interactions = []
                    for rep_i in range(n_reps):
                        ref = emg_avg[rep_i]
                        interactions.append(ref[i,j])

                    interactions = np.asarray(interactions)
                    avg_interaction = np.mean(interactions, axis=0)
                    print(f"subject {m_i}, emg {e_i} | {problem['names'][i]} & {problem['names'][j]}: S2 = {avg_interaction:.4f}")
            
        s2 = np.array(s2)
        
        S2.append(np.mean(s2, axis=0))

    overall_avg = np.mean(np.array(S2), axis=0)
    print("\nAvg second-order interaction terms:")
    for i in range(D):
        for j in range(i+1, D):
            print(f"x{i} & x{j}: Avg S2 = {overall_avg[i,j]:.4f}")


### --- BO methods --- ###

def run_single_neurostim(subject_idx, emg_idx, kappa_idx, kappa, rep_idx,
                          dataset, model_cls, device, options, lr=0.1, M=1024):
    """
    Worker function for a single (subject, emg, kappa, rep) job.

    Returns dict with:
      - indices: (subject_idx, emg_idx, kappa_idx, rep_idx)
      - perf_explore: array[MaxQueries]
      - perf_exploit: array[MaxQueries]
      - rsq: float
      - regret: array[MaxQueries]
      - sobol_interactions: array[MaxQueries] of objects
      - train_time: array[MaxQueries]
      - cum_time: array[MaxQueries]
      - model_name: str
    """
    device = torch.device(device)
    nrnd = options['n_rnd']
    MaxQueries = options['n_queries']
    ndims = options['n_dims']

    print(f'[{device}] subject {subject_idx + 1} | emg: {emg_idx} | kappa {kappa} | rep {rep_idx + 1}')

    try:
        # Load subject data inside worker to avoid serialization issues
        subject = load_data(dataset, subject_idx)
        subject['ch2xy'] = torch.tensor(subject['ch2xy'], device=device)

        # "Ground truth" map
        MPm = torch.tensor(subject['sorted_respMean'][:, emg_idx], device=device).float()
        # Best known channel
        mMPm = torch.max(MPm)

        # priors and kernel handling
        priorbox = gpytorch.priors.SmoothedBoxPrior(
            a=math.log(options['rho_low']), b=math.log(options['rho_high']), sigma=0.01)
        outputscale_priorbox = gpytorch.priors.SmoothedBoxPrior(
            a=math.log(0.01**2), b=math.log(100.0**2), sigma=0.01)

        prior_lik = gpytorch.priors.SmoothedBoxPrior(
            a=options['noise_min']**2, b=options['noise_max']**2, sigma=0.01)
        likf = gpytorch.likelihoods.GaussianLikelihood(noise_prior=prior_lik)
        likf.initialize(noise=torch.tensor(1.0, device=device, dtype=torch.get_default_dtype()))

        if 'cuda' in str(device):
            likf = likf.cuda()

        # Metrics initialization
        DimSearchSpace = subject['DimSearchSpace']
        P_test = torch.zeros((MaxQueries, 2), device=device)
        train_time = np.zeros(MaxQueries, dtype=np.float32)
        cum_time = np.zeros(MaxQueries, dtype=np.float32)
        regret = np.zeros(MaxQueries, dtype=np.float32)
        sobol_interactions = np.empty(MaxQueries, dtype=object)

        # maximum response obtained in this round
        MaxSeenResp = 0
        q = 0
        timer = 0.0
        order_this = torch.randperm(DimSearchSpace, device=device)
        P_max = []

        executor = ThreadPoolExecutor(max_workers=4)
        space_reconfiguration = None
        interactions = None
        model = None
        sobol = None

        while q < MaxQueries:
            t0 = time.time()
            # Query selection
            if q >= nrnd:
                # Max of acquisition map
                AcquisitionMap = ymu + kappa * torch.nan_to_num(torch.sqrt(ys2))
                Next_Elec = torch.where(
                    AcquisitionMap.reshape(len(AcquisitionMap)) == torch.max(AcquisitionMap.reshape(len(AcquisitionMap))))
                Next_Elec = Next_Elec[0][np.random.randint(len(Next_Elec))] if len(Next_Elec[0]) > 1 else Next_Elec[0][0]
                P_test[q][0] = Next_Elec
            else:
                P_test[q][0] = int(order_this[q])
            query_elec = P_test[q][0]

            # Read response
            sample_resp = torch.tensor(
                subject['sorted_resp'][int(query_elec)][emg_idx][subject['sorted_isvalid'][int(query_elec)][emg_idx] != 0])
            if len(sample_resp) == 0:
                sample_resp = torch.tensor(subject['sorted_resp'][int(query_elec)][emg_idx])
                valid_responses = subject['sorted_isvalid'][int(query_elec)][emg_idx]
                print(f'sample_response: {sample_resp} \n valid_responses: {valid_responses} \n and query_elec: {query_elec}\n')
                test_respo = 1e-9
            else:
                test_respo = sample_resp[np.random.randint(len(sample_resp))]

            std = (0.02 * torch.mean(test_respo)).clamp(min=0.0)
            noise = torch.randn((), device=test_respo.device, dtype=test_respo.dtype) * std
            test_respo = test_respo + noise
            P_test[q][1] = test_respo

            if (test_respo > MaxSeenResp) or (MaxSeenResp == 0):
                MaxSeenResp = test_respo

            x = subject['ch2xy'][P_test[:q + 1, 0].long(), :].float()
            x = x.reshape((len(x), ndims))
            y = P_test[:q + 1, 1] / MaxSeenResp

            # Model initialization and model update
            if q == 0:
                model = model_cls(x, y, likf, priorbox, outputscale_priorbox)
                if model.name == 'NeuralSobolGP':
                    sobol = NeuralSobol(dataset, M=M).to(device)
                    model.sobol = sobol
                    interactions = np.zeros((ndims), dtype=float)
            else:
                model.set_train_data(x, y, strict=False)

            # Model training
            model.train()
            likf.train()
            model.to(device), likf.to(device)
            model, likf, _ = optimize(model, x.to(device), y.to(device), lr=lr)

            # Model evaluation
            model.eval()
            likf.eval()

            with torch.no_grad():
                X_test = subject['ch2xy'].float()
                observed_pred = likf(model(X_test))
                ymu = observed_pred.mean
                ys2 = observed_pred.variance

            Tested = torch.unique(P_test[:q + 1, 0]).long()
            MapPredictionTested = ymu[Tested]
            BestQuery = Tested if len(Tested) == 1 else Tested[
                (MapPredictionTested == torch.max(MapPredictionTested)).reshape(len(MapPredictionTested))]
            if len(BestQuery) > 1:
                BestQuery = np.array([BestQuery[np.random.randint(len(BestQuery))].cpu()])

            # Store metrics
            P_max.append(BestQuery[0])

            # log regret calculations
            chosen_pref = MPm[BestQuery].item()
            instant_regret = mMPm.item() - chosen_pref
            regret[q] = instant_regret

            # Pending Sobol job fetch
            if space_reconfiguration is not None and space_reconfiguration.done():
                interactions = space_reconfiguration.result()
                new_partition = sobol.update_partition(interactions)
                model.update_partition(new_partition)
                space_reconfiguration = None

            # Update partitions
            if model.name == 'NeuralSobolGP':
                sobol_interactions[q] = interactions.copy()

                if (space_reconfiguration is None or space_reconfiguration.done()) and (q > (MaxQueries // 4)):
                    surrogate_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=prior_lik).to(device)
                    surrogate = NeuralExactGP(x, y, surrogate_likelihood, priorbox, outputscale_priorbox).to(device)
                    space_reconfiguration = executor.submit(sobol.method, x, y, surrogate)

            # computation time calculations
            t1 = time.time()
            elapsed = t1 - t0
            timer += elapsed
            train_time[q] = elapsed
            cum_time[q] = timer
            q += 1

        # estimate current exploration performance
        perf_explore = (MPm[P_max].reshape((len(MPm[P_max]))) / mMPm).cpu().numpy()
        # estimate current exploitation performance
        perf_exploit = P_test[:, 0].long().cpu().numpy()
        # R^2 correlation with ground truth
        y_true = MPm
        y_pred = ymu

        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r_squared = (1 - (ss_res / ss_tot)).item()

        model_name = model.name

        # Storage Cleanup
        executor.shutdown(wait=True)
        del model
        del likf
        del observed_pred
        del ymu
        del ys2
        del X_test
        del subject 
        del MPm
        
        if 'cuda' in str(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        gc.collect()

        return {
            'indices': (subject_idx, emg_idx, kappa_idx, rep_idx),
            'perf_explore': perf_explore,
            'perf_exploit': perf_exploit,
            'rsq': r_squared,
            'regret': regret,
            'sobol_interactions': sobol_interactions,
            'train_time': train_time,
            'cum_time': cum_time,
            'model_name': model_name
        }

    except Exception as e:
        print(f"FAILED: subject={subject_idx} emg={emg_idx} kappa={kappa} rep={rep_idx}: {e}")
        traceback.print_exc()
        return None

def neurostim_bo(dataset, model_cls, kappas, devices=['cpu'], lr=0.1, M=2048):
    """
    Run neurostimulation Bayesian optimization across subjects, EMGs, kappas, and reps.
    Uses ProcessPoolExecutor to distribute work across multiple devices.

    Parameters
    ----------
    dataset : str
        Dataset identifier (e.g., '5d_rat', 'nhp')
    model_cls : class
        GP model class (NeuralExactGP, NeuralAdditiveGP, NeuralSobolGP)
    kappas : list of float
        UCB exploration coefficients
    devices : list of str
        List of devices for parallel execution (e.g., ['cpu'], ['cuda:0', 'cuda:1'])
    lr : float
        Learning rate for GP optimization (default 0.1)
    M : int
        Sobol MC samples (default 1024)
    """
    np.random.seed(0)

    # Experiment parameters initialization
    options = set_experiment(dataset)
    nRep = options['n_reps']
    nSubjects = options['n_subjects']
    nEmgs = options['n_emgs']
    MaxQueries = options['n_queries']

    # Metrics initialization (CPU numpy arrays)
    PP = np.zeros((nSubjects, max(nEmgs), len(kappas), nRep, MaxQueries), dtype=np.float32)
    PP_t = np.zeros((nSubjects, max(nEmgs), len(kappas), nRep, MaxQueries), dtype=np.float32)
    Q = np.zeros((nSubjects, max(nEmgs), len(kappas), nRep, MaxQueries), dtype=np.float32)
    Train_time = np.zeros((nSubjects, max(nEmgs), len(kappas), nRep, MaxQueries), dtype=np.float32)
    Cum_train = np.zeros((nSubjects, max(nEmgs), len(kappas), nRep, MaxQueries), dtype=np.float32)
    RSQ = np.zeros((nSubjects, max(nEmgs), len(kappas), nRep), dtype=np.float32)
    REGRETS = np.zeros((nSubjects, max(nEmgs), len(kappas), nRep, MaxQueries), dtype=np.float32)
    SOBOLS = np.empty((nSubjects, max(nEmgs), len(kappas), nRep, MaxQueries), dtype=object)

    # Build job list with round-robin device assignment
    jobs = []
    job_idx = 0
    num_workers = len(devices)

    for s_idx in range(nSubjects):
        # Load subject to get number of EMGs
        n_emgs_this_subject = options['n_emgs'][s_idx]

        for k_idx, kappa in enumerate(kappas):
            for e_i in range(n_emgs_this_subject):
                for rep_i in range(nRep):
                    device = devices[job_idx % num_workers]
                    jobs.append({
                        'subject_idx': s_idx,
                        'emg_idx': e_i,
                        'kappa_idx': k_idx,
                        'kappa': kappa,
                        'rep_idx': rep_i,
                        'dataset': dataset,
                        'model_cls': model_cls,
                        'device': device,
                        'options': options,
                        'lr': lr,
                        'M': M,
                    })
                    job_idx += 1

    print(f"\nStarting Neurostim BO with {len(jobs)} total jobs on {num_workers} workers: {devices}")

    # Execute jobs in parallel
    model_name = 'Unknown'
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_neurostim, **job) for job in jobs]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                s, e, k, r = result['indices']
                PP[s, e, k, r] = result['perf_explore']

                # Load subject to get MPm for exploitation conversion
                subject = load_data(dataset, s)
                MPm = torch.tensor(subject['sorted_respMean'][:, e]).float()
                mMPm = torch.max(MPm)
                PP_t[s, e, k, r] = (MPm[result['perf_exploit'].astype(int)] / mMPm).numpy()

                Q[s, e, k, r] = result['perf_exploit']
                Train_time[s, e, k, r] = result['train_time']
                Cum_train[s, e, k, r] = result['cum_time']
                RSQ[s, e, k, r] = result['rsq']
                REGRETS[s, e, k, r] = np.log(result['regret'] + 1e-8)
                SOBOLS[s, e, k, r] = result['sobol_interactions']
                model_name = result['model_name']

    # Saving variables
    output_dir = os.path.join('output', 'neurostim_experiments', dataset)
    os.makedirs(output_dir, exist_ok=True)
    fname = f'{dataset}_{model_name}_budget{MaxQueries}_{nRep}reps.npz'
    results_path = os.path.join(output_dir, fname)
    np.savez_compressed(results_path,
                        RSQ=RSQ, PP=PP, PP_t=PP_t,
                        kappas=np.array(kappas),
                        SOBOLS=SOBOLS,
                        REGRETS=REGRETS,
                        Train_time=Train_time,
                        Cum_train=Cum_train)
    print(f'saved results to {results_path}')

    return {
        'PP': PP, 'PP_t': PP_t, 'REGRETS': REGRETS,
        'RSQ': RSQ, 'Train_time': Train_time, 'Cum_train': Cum_train,
        'SOBOLS': SOBOLS, 'model_name': model_name,
    }

def lr_search(dataset, lr_list, kappa, devices=['cpu'], M=2048):
    """
    Run learning rate search for all 3 models (NeuralExactGP, NeuralAdditiveGP, NeuralSobolGP).

    For each model and each lr, runs neurostim_bo and aggregates exploration/log-regret curves.
    Produces a combined 3-row x 2-column plot (one row per model).

    Parameters
    ----------
    dataset : str
        Dataset identifier (e.g., 'nhp', 'rat')
    lr_list : list of float
        Learning rates to sweep
    kappa : float
        Single kappa for all runs
    devices : list of str
    M : int
        Sobol MC samples
    """
    from models.gaussians import NeuralExactGP, NeuralAdditiveGP, NeuralSobolGP

    model_specs = [
        (NeuralExactGP, 'NeuralExactGP'),
        (NeuralAdditiveGP, 'NeuralAdditiveGP'),
        (NeuralSobolGP, 'NeuralSobolGP'),
    ]

    all_results = {}  # {model_name: {lr: {'PP': ..., 'REGRETS': ...}}}

    for model_cls, model_name in model_specs:
        all_results[model_name] = {}
        for lr in lr_list:
            print(f"\n--- lr_search: {model_name} lr={lr} kappa={kappa} ---")
            res = neurostim_bo(dataset, model_cls, kappas=[kappa],
                               devices=devices, lr=lr, M=M)
            all_results[model_name][lr] = res

    # Plotting: 3 rows (one per model), 2 columns (exploration, log-regret)
    n_models = len(model_specs)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models), squeeze=False)
    cmap = plt.get_cmap('tab10')

    for row, (model_cls, model_name) in enumerate(model_specs):
        ax_explore = axes[row, 0]
        ax_regret = axes[row, 1]

        for idx, lr in enumerate(lr_list):
            res = all_results[model_name][lr]
            # PP shape: (nSubjects, max(nEmgs), len(kappas), nRep, MaxQueries)
            # Average over subjects, emgs, kappas (dim=1), reps -> (MaxQueries,)
            pp = res['PP']
            regrets = res['REGRETS']

            # Mean over all non-query dims: subjects(0), emgs(1), kappas(2), reps(3)
            mean_explore = np.nanmean(pp, axis=(0, 1, 2, 3))
            mean_regret = np.nanmean(regrets, axis=(0, 1, 2, 3))

            color = cmap(idx % 10)
            x = np.arange(len(mean_explore))

            ax_explore.plot(x, mean_explore, color=color, label=f'lr={lr}')
            ax_regret.plot(x, mean_regret, color=color, label=f'lr={lr}')

        ax_explore.set_title(f'{model_name} - Exploration')
        ax_explore.set_xlabel('Query')
        ax_explore.set_ylabel('Exploration Score')
        ax_explore.set_ylim(0, 1.1)
        ax_explore.legend(loc='lower right', fontsize='small')
        ax_explore.grid(True)

        ax_regret.set_title(f'{model_name} - Log Regret')
        ax_regret.set_xlabel('Query')
        ax_regret.set_ylabel('Log Regret')
        ax_regret.legend(loc='upper right', fontsize='small')
        ax_regret.grid(True)

    fig.suptitle(f'LR Search on {dataset} (kappa={kappa})', fontsize=14)
    plt.tight_layout()

    output_dir = os.path.join('output', 'neurostim_experiments', dataset)
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'lr_search_{dataset}.svg')
    plt.savefig(plot_path, format='svg')
    plt.close(fig)
    print(f"Saved lr_search plot to {plot_path}")

    return all_results

### --- Parser handling --- ###

def _parse_list_of_floats(text):
    if text is None:
        return None
    try:
        return [float(x) for x in text.split(',') if len(x.strip())]
    except Exception:
        raise argparse.ArgumentTypeError('Expected comma-separated list of numbers (e.g. 0.5,1,3)')

def _parse_list_of_ints(text):
    if text is None:
        return None
    try:
        return [int(x) for x in text.split(',') if len(x.strip())]
    except Exception:
        raise argparse.ArgumentTypeError('Expected comma-separated list of ints (e.g. 1,2,3)')

def _parse_list_of_strings(text):
    if text is None:
        return None
    try:
        return [x.strip() for x in text.split(',') if len(x.strip())]
    except Exception:
        raise argparse.ArgumentTypeError('Expected comma-separated list of strings')

def main(argv=None):
    parser = argparse.ArgumentParser(description='Neurostimulation Bayesian Optimization runner with CLI options')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='5d_rat', help='Neurostimulation dataset type')

    # Method selection
    parser.add_argument('--method', type=str, default='neurostim_bo',
                        help='Method: neurostim_bo, lr_search, M_search')

    # Model selection
    parser.add_argument('--model_cls', type=str, default='NeuralExactGP', help='Model class name (NeuralExactGP, NeuralAdditiveGP, NeuralSobolGP)')

    # Method-specific params
    parser.add_argument('--kappas', type=_parse_list_of_floats, default=None, help='Comma-separated kappas for experiments')
    parser.add_argument('--kappa', type=float, default=None, help='Single kappa for hyperparam searches (lr_search, M_search)')
    parser.add_argument('--lr', type=float, default=0.1, help='GP learning rate (default 0.1)')
    parser.add_argument('--M', type=int, default=1024, help='Sobol MC samples (default 1024)')
    parser.add_argument('--lr_list', type=_parse_list_of_floats, default=None, help='Comma-separated learning rates for lr_search')
    parser.add_argument('--M_list', type=_parse_list_of_ints, default=None, help='Comma-separated M values for M_search')

    # Device selection
    parser.add_argument('--device', type=str, default='cpu', help='Device for computation (e.g., cpu, cuda:0)')
    parser.add_argument('--devices', type=_parse_list_of_strings, default=None, help='Comma-separated devices for parallel execution (e.g., cuda:0,cuda:1,cpu)')

    # Misc
    parser.add_argument('--list_models', action='store_true')
    parser.add_argument('--list_methods', action='store_true')

    args = parser.parse_args(argv)

    # Allowed mappings (whitelist)
    model_map = {
        'NeuralExactGP': NeuralExactGP,
        'NeuralAdditiveGP': NeuralAdditiveGP,
        'NeuralSobolGP': NeuralSobolGP,
    }

    method_map = {
        'neurostim_bo': neurostim_bo,
        'lr_search': lr_search,
        'M_search': M_search_neurostim,
    }

    if args.list_models:
        print('Available model classes:')
        for k in model_map.keys():
            print(' -', k)
        return

    if args.list_methods:
        print('Available methods:')
        for k in method_map.keys():
            print(' -', k)
        return

    if args.model_cls not in model_map:
        raise ValueError(f"Unknown model_cls '{args.model_cls}'. Use --list_models to see options.")

    if args.method not in method_map:
        raise ValueError(f"Unknown method '{args.method}'. Use --list_methods to see options.")

    model_cls = model_map[args.model_cls]

    # Determine device list
    if args.devices:
        device_list = args.devices
    else:
        device_list = [args.device]

    # dispatch
    try:
        if args.method == 'neurostim_bo':
            result = neurostim_bo(args.dataset, model_cls, kappas=args.kappas,
                                  devices=device_list, lr=args.lr, M=args.M)

        elif args.method == 'lr_search':
            if args.kappa is None:
                raise ValueError('--kappa must be provided for lr_search')
            lr_list_val = args.lr_list or [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
            result = lr_search(args.dataset, lr_list_val, kappa=args.kappa,
                               devices=device_list, M=args.M)

        elif args.method == 'M_search':
            if args.kappa is None:
                raise ValueError('--kappa must be provided for M_search')
            m_list_val = args.M_list or [256, 512, 1024, 2048]
            result = M_search_neurostim(args.dataset, m_list_val, kappa=args.kappa,
                                        devices=device_list, lr=args.lr)

        else:
            raise ValueError(f'Unsupported method: {args.method}')

        print('Completed')

    except Exception as e:
        print('ERROR during execution:', e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':

    main()
    #neurostim_bo('spinal', NeuralSobolGP, kappas=[25.0], devices=['cpu', 'cuda:0', 'cuda:1'] )