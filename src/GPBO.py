import numpy as np
import matplotlib.pyplot as plt
import textwrap
import sys
import os
import torch
import gpytorch
import datetime
import random
import copy
import argparse
import time
from gpytorch.models.exact_gp import GPInputWarning

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from SALib.analyze import sobol as salib_sobol
from SALib.sample import saltelli
from SALib.sample import sobol as sobol_sampler
from scipy.stats import qmc, uniform
from scipy.stats import sobol_indices as scipy_sobol


from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
import traceback

### MODULES HANDLING ###
from utils.synthetic_datasets import *
from utils.sensitivity_utils import *
from models.gaussians import *
from models.sobols import *

warnings.filterwarnings("ignore", category=FutureWarning, module="SALib.util")


### Runners ###

def run_partitionbo(f_obj,  model_cls=SobolGP, n_iter=200, n_sobol=20, kappa=1.0, save=False, verbose=False,  device='cpu'):
    """
    Returns the same metrics as run_bo, plus:
      - partition_updates: list of partition structures (list of lists) when updates occurred
      - sobol_interactions: list of interaction matrices (numpy arrays) when updates occurred
    """

    # Setting device for computations
    device = torch.device(device)

    sobol_workers = 1   # background threads for Sobol
    # --------------------------------------------------------

    # Initiate n_init points, bounds
    n_points = int(100 * f_obj.d**2)
    bounds = torch.from_numpy(np.stack([f_obj.lower_bounds, f_obj.upper_bounds])).float().to(device)  # shape (2, d)
    lower, upper = bounds[0], bounds[1]
    grid_x = lower + (upper - lower) * torch.rand(n_points, f_obj.d, device=device)
    with torch.no_grad():
        grid_y = f_obj.f.forward(grid_x).float().squeeze(-1) 
    grid_min = grid_y.min().item()

    # Initiate training set
    sobol = Sobol(f_obj).to(device)
    n_init = f_obj.d*3
    true_best = grid_y.max().item() #f_obj.f.optimal_value
    train_x_cpu = torch.from_numpy(sobol_sampler.sample(sobol.problem, n_init, calc_second_order=False)).float()
    train_x = train_x_cpu.to(device)
    train_y = f_obj.f.forward(train_x)
    best_observed = train_y.max().item()

    current_min = min(grid_min, train_y.min().item())
    current_max = max(true_best, train_y.max().item())

    # initialize metrics (same as run_bo)
    r_squared, loss_curve, expl_scores, exploit_scores, regrets, train_times = [0.0 for i in range(n_init)], [], [0.0 for i in range(n_init)], [0.0 for i in range(n_init)], [1.0 for i in range(n_init)], [0.0 for i in range(n_init)]

    # Additional metrics
    partition_updates = []
    sobol_interactions = []

    # Initialize executor
    executor = ThreadPoolExecutor(max_workers=sobol_workers)
    space_reconfiguration = None

    # initialize model 
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    if model_cls.__name__ == 'MHGP':
        with torch.no_grad():
            likelihood.noise = torch.tensor(1e-3)
    model = model_cls(train_x, train_y, likelihood).to(device)
    model.sobol = sobol
    interactions = None
    # main optimization loop
    for i in range(n_iter - n_init):

        # Train model
        t0 = time.time()
        y_range = current_max - current_min
        train_y_norm = (train_y - current_min) / y_range
        model, likelihood, epoch_losses = optimize(model, train_x, train_y_norm)
        loss_curve.append(np.mean(epoch_losses))

        new_x, acq_val, acq_idx = maximize_acq(kappa, model, likelihood, grid_x)
        new_y = f_obj.f.forward(new_x)

        # Update global bounds and exploration metrics
        new_y_val = new_y.item()
        if new_y_val < current_min:
            current_min = new_y_val
        if new_y_val > current_max:
            current_max = new_y_val
            
        # Keep your original tracking for the final score
        if new_y_val > true_best:
            true_best = new_y_val
        if new_y_val < grid_min:
            grid_min = new_y_val
        
        train_x = torch.cat([train_x, new_x])
        train_y  = torch.cat([train_y, new_y])
        
        # Posterior predictions for R^2
        model.eval(); likelihood.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning if hasattr(gpytorch.utils, "warnings") else Warning)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                post = likelihood(model(grid_x))

            y_true = ((grid_y.squeeze(-1) - grid_min) / (true_best - grid_min)).cpu()
            y_pred = post.mean.cpu()
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
            r2_score = np.clip(1 - ss_res / ss_tot, 0.0, 1.0)
            r_squared.append(r2_score.item())

        # Exploration and Exploitation scores (normalized)
        best_observed = train_y.max().item()
        explore = (best_observed - grid_min) / (true_best - grid_min + 1e-9)
        exploit = (new_y.item() - grid_min) / (true_best - grid_min + 1e-9)
        explore = np.clip(explore, 0.0, 1.0)
        exploit = np.clip(exploit, 0.0, 1.0)
        exploit_scores.append(exploit.item())
        expl_scores.append(explore.item())

        # Regret scores
        regret = true_best - best_observed + 1e-9
        regrets.append(regret)

        # If there is a pending Sobol job finished, fetch it and update partition
        if space_reconfiguration is not None and space_reconfiguration.done():
            interactions = space_reconfiguration.result()
            if model.name == 'MHGP':
                new_partition = model.metropolis_hastings(interactions)
            else:
                new_partition = sobol.update_partition(interactions)
            model.update_partition(new_partition)
            sobol_interactions.append(interactions)
            partition_updates.append(new_partition)
            space_reconfiguration = None
        else:
            sobol_interactions.append(interactions)
            partition_updates.append(model.partition)
            
        t1 = time.time()
        elapsed_train = t1 - t0
        train_times.append(elapsed_train)
        model = model_cls(train_x, train_y_norm, likelihood,
                          model.partition, model.history, model.sobol).to(device) 
        

        # Submit a new Sobol background job every n_sobol iterations (if none pending)
        if (i % n_sobol) == 0:
            if space_reconfiguration is None or space_reconfiguration.done():
                try:
                    surrogate_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
                    y_range = current_max - current_min
                    train_y_norm = (train_y - current_min) / y_range
                    surrogate = ExactGP(train_x, train_y_norm, surrogate_likelihood).to(device)
                    space_reconfiguration = executor.submit(sobol.method, train_x, train_y_norm, surrogate)
                except Exception as e:
                    space_reconfiguration = None
                    print(f"[Sobol] submit failed: {e}")

        if verbose:
            print(f"Iter {i+1:02d} Loss: {epoch_losses:.4f} Exploit score: {exploit:.4f}, Explore score: {explore:.4f}, R^2: {r2_score:.4f} | partition: {model.partition}")
            print(f'best observed: {best_observed:.4f} |  true best: {true_best:.4f} | current query: {new_y.item():.4f}')

    executor.shutdown(wait=False)

    # compute cumulative training time per-iteration
    train_times = np.array(train_times, dtype=np.float64)
    cumulative_time = np.cumsum(train_times)
    # Save if requested (similar layout as run_bo, but with additional fields)
    if save:
        output_dir = os.path.join("output", "breaking_additivity", "sobolbo")
        os.makedirs(output_dir, exist_ok=True)
        fname = f"SobolBO_{datetime.date.today().isoformat()}_kappa{kappa}_{f_obj.name}.npz"
        fpath = os.path.join(output_dir, fname)
        # convert partition_updates to object array for saving
        np.savez(fpath,
                 r_squared=np.array(r_squared),
                 loss_curve=np.array(loss_curve),
                 exploration=np.array(expl_scores),
                 exploitation=np.array(exploit_scores),
                 regrets=np.array(regrets),
                 partition_updates=np.array(partition_updates, dtype=object),
                 sobol_interactions=np.array(sobol_interactions, dtype=object))

    # final packaging of results
    result = {
        'r2': np.array(r_squared),
        'exploration': np.array(expl_scores),
        'exploitation': np.array(exploit_scores),
        'loss_curve': np.array(loss_curve),
        'regrets': np.array(regrets),
        'partition_updates': partition_updates,
        'sobol_interactions': sobol_interactions,
        'train_times': train_times,        
        'cumulative_time': cumulative_time 
    }
    return result

def run_bo(f_obj, model_cls, n_iter=200, kappa=1.0, save=False, verbose=False, device='cpu'):

    # Setting device for computations
    device = torch.device(device)

    # Initiate n_init points, bounds
    n_points = int(100 * f_obj.d**2)
    bounds = torch.from_numpy(np.stack([f_obj.lower_bounds, f_obj.upper_bounds])).float().to(device)  # shape (2, d)
    lower, upper = bounds[0], bounds[1]
    grid_x = lower + (upper - lower) * torch.rand(n_points, f_obj.d, device=device)
    with torch.no_grad():
        grid_y = f_obj.f.forward(grid_x).float().squeeze(-1) 
    grid_min = grid_y.min().item()
    
    # Initiate training set
    sobol = Sobol(f_obj)
    n_init = f_obj.d*3
    true_best = grid_y.max().item() #f_obj.f.optimal_value  
    train_x_cpu = torch.from_numpy(sobol_sampler.sample(sobol.problem, n_init, calc_second_order=False)).float()
    train_x = train_x_cpu.to(device)
    train_y = f_obj.f.forward(train_x)
    best_observed = train_y.max().item()

    current_min = min(grid_min, train_y.min().item())
    current_max = max(true_best, train_y.max().item())

    #initialize metrics
    r_squared, loss_curve, expl_scores, exploit_scores, regrets, train_times = [0.0 for i in range(n_init)], [], [0.0 for i in range(n_init)], [0.0 for i in range(n_init)], [1.0 for i in range(n_init)], [0.0 for i in range(n_init)]
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = model_cls(train_x, train_y, likelihood).to(device)

    for i in range(n_iter - n_init):

        # Train model
        t0 = time.time()
        y_range = current_max - current_min
        train_y_norm = (train_y - current_min) / y_range
        model, likelihood, epoch_losses = optimize(model, train_x, train_y_norm)

        loss_curve.append(np.mean(epoch_losses))
        new_x, acq_val, acq_idx = maximize_acq(kappa, model, likelihood, grid_x)

        new_y = f_obj.f.forward(new_x)

        # Update global bounds and exploration metrics
        new_y_val = new_y.item()
        if new_y_val < current_min:
            current_min = new_y_val
        if new_y_val > current_max:
            current_max = new_y_val
            
        # Keep your original tracking for the final score
        if new_y_val > true_best:
            true_best = new_y_val
        if new_y_val < grid_min:
            grid_min = new_y_val
        
        train_x = torch.cat([train_x, new_x])
        train_y  = torch.cat([train_y, new_y])

        # Posterior predictions for R^2
        model.eval(); likelihood.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=GPInputWarning)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                post = likelihood(model(grid_x))
            
            y_true = ((grid_y.squeeze(-1) - grid_min) / (true_best - grid_min)).cpu()
            y_pred = post.mean.cpu()
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
            r2_score = np.clip(1 - ss_res / ss_tot, 0.0, 1.0)
            r_squared.append(r2_score.item())

        # Exploration and Exploitation scores (normalized) 
        best_observed = train_y.max().item()
        explore = (best_observed - grid_min) / (true_best - grid_min + 1e-9)
        exploit = (new_y.item() - grid_min) / (true_best - grid_min + 1e-9)
        explore = np.clip(explore, 0.0, 1.0)
        exploit = np.clip(exploit, 0.0, 1.0)
        exploit_scores.append(exploit.item())
        expl_scores.append(explore.item())

        # Regret scores
        regret = true_best - best_observed + 1e-9
        regrets.append(regret)

        model = model_cls(train_x, train_y, likelihood).to(device)

        t1 = time.time()
        elapsed_train = t1 - t0
        train_times.append(elapsed_train)

        if verbose:
            print(f"{model.name} | Iter {i+1:02d} Exploit score: {exploit:.2f}, Explore score: {explore:.3f}, R^2: {r2_score}")
            print(f'best observed: {best_observed:.2f} |  true best: {true_best:.2f} | current query: {new_y.item():.2f}')

    train_times = np.array(train_times, dtype=np.float64)
    cumulative_time = np.cumsum(train_times)

    if save:
    # Save results
        output_dir = os.path.join("output", "breaking_additivity")
        fname = f"{model.name}_{datetime.date.today().isoformat()}_kappa{kappa}_{f_obj.name}.npz"
        fpath = os.path.join(output_dir, fname)

        np.savez(fpath,
                r_squared=np.array(r_squared),
                loss_curve=np.array(loss_curve),
                exploration=np.array(expl_scores),
                exploitation=np.array(exploit_scores),
                regrets=np.array(regrets))
        
    r_squared=np.array(r_squared)
    loss_curve=np.array(loss_curve)
    exploration=np.array(expl_scores)
    exploitation=np.array(exploit_scores)
    regrets=np.array(regrets)

    return {
        'r2': r_squared,
        'exploration': exploration,
        'exploitation': exploitation,
        'loss': loss_curve,
        'regrets': regrets,
        'train_times': train_times,        
        'cumulative_time': cumulative_time 
       }

def run_single_kappa(kappa, device, f_obj, model_cls, n_iter, n_reps, bo_method):
    """
    Worker function that runs the repetitions for a SINGLE kappa on a specific DEVICE.
    """
    explore_list, exploit_list = [], []
    
    f_obj.to(device)

    print(f"[Worker {device}] Starting kappa={kappa}")
    
    for rep in range(n_reps):
        try:
            results = bo_method(f_obj, model_cls, n_iter=n_iter, 
                                kappa=kappa, save=False, device=device)
            explore_list.append(results['exploration'])
            exploit_list.append(results['exploitation'])
        except Exception as e:
            print(f"Error in kappa={kappa}, rep={rep} on {device}: {e}")
            traceback.print_exc()
            # Append zeros or handle error gracefully to avoid crashing the whole pool
            explore_list.append(np.zeros(n_iter))
            exploit_list.append(np.zeros(n_iter))

    stacked_explore = np.stack(explore_list, axis=0)
    stacked_exploit = np.stack(exploit_list, axis=0)
    
    return kappa, stacked_explore.mean(axis=0), stacked_exploit.mean(axis=0)

def run_single_optm(model_label, model_cls, kappa, rep, device, 
                           f_obj, n_iter):
    """
    Worker to run a single repetition for a specific model class.
    """
    f_obj.to(device)
        
    print(f"[Worker {device}] Starting {model_label} rep {rep+1} (k={kappa})")
    
    try:
        # Select correct runner
        if model_label in ['SobolGP', 'MHGP']:
            res = run_partitionbo(f_obj, model_cls, n_iter=n_iter, 
                                  kappa=kappa, save=False, device=device)
        else:
            res = run_bo(f_obj, model_cls, n_iter=n_iter, 
                         kappa=kappa, save=False, device=device)
            
        # Return lightweight data (numpy arrays)
        return (model_label, {
            'regrets': res['regrets'],
            'exploration': res['exploration'],
            'exploitation': res['exploitation'],
            'r2': res['r2']
        })
    except Exception as e:
        print(f"FAILED: {model_label} rep {rep} on {device}: {e}")
        traceback.print_exc()
        return None

### Graph generation functions ###

def kappa_search(f_obj, kappa_list, model_cls=ExactGP, n_iter=100, n_reps=15,
                bo_method=run_bo, devices=['cpu']):
    """
    For each kappa in kappa_list, run BO n_reps times and plot averaged exploration
    (dashed line) and exploitation (filled area) traces. The x-axis begins with
    `n_init` zeros (initialization period) followed by the BO iterations.

    - exploration: dashed line
    - exploitation: filled area (solid edge + alpha fill)
    - each kappa uses a distinct color

    Returns:
      dict with averaged traces per kappa and path to saved plot.
    """

    averaged = {}
    n_init = 3*f_obj.d

    # 1. Setup Parallel Workers
    # If we have more kappas than devices, we cycle through devices.
    # Logic: Create a list of tasks where each task is assigned a device.
    
    num_workers = len(devices)
    print(f"\nStarting Kappa Search with {num_workers} workers on devices: {devices}")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_kappa = {}
        
        for i, kappa in enumerate(kappa_list):
            # Round-robin assignment of devices
            assigned_device = devices[i % num_workers]
            
            future = executor.submit(
                run_single_kappa,
                kappa, 
                assigned_device, 
                f_obj, 
                model_cls, 
                n_init, 
                n_iter, 
                n_reps, 
                bo_method, 
            )
            future_to_kappa[future] = kappa

        # 2. Collect Results as they finish
        for future in as_completed(future_to_kappa):
            kappa = future_to_kappa[future]
            try:
                k_res, mean_explore, mean_exploit = future.result()
                averaged[k_res] = {'explore': mean_explore, 'exploit': mean_exploit}
                print(f"Finished kappa={k_res}")
            except Exception as exc:
                print(f'Kappa {kappa} generated an exception: {exc}')


    # Plotting
    print("\nAll runs complete. Generating Plot...")
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    x = np.arange(0, n_iter)

    for idx, kappa in enumerate(kappa_list):

        vals = averaged.get(kappa, None)
        color = cmap(idx % 10)

        mean_explore = vals['explore']
        mean_exploit = vals['exploit']

        # plot exploration as dashed line
        plt.plot(x, mean_explore, linestyle='-', marker=None, label=f'k={kappa} Explore', color=color)
        plt.plot(x, mean_exploit, linestyle='--', marker=None, color=color)


    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.title(f"{model_cls.__name__} on {f_obj.d}-{f_obj.name} | Exploration & Exploitation curves (averaged over {n_reps} runs)")
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save plot
    output_dir = os.path.join('output', 'synthetic_experiments', f_obj.name)
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f'kappa_search_{model_cls.__name__}_{f_obj.f.d}-d_budget{n_iter}.svg'
    )
    plt.savefig(plot_path, format="svg")
    plt.close()
    print(f"Saved kappa search plot to {plot_path}")

def optimization_metrics(f_obj, kappas, n_init=6, n_iter=100, n_reps=15, ci=95, devices=['cpu']):
    """
    Run BO variants for f_obj and plot:
      - Plot 1: Exploration (dashed), Exploitation (solid) averaged across n_reps
                    for ExactGP (red), AdditiveGP (blue), SobolGP (green), MHGP (orange)
      - Plot 2: Mean regret +/- CI, plotted on a log y-scale (log-regrets)
    Results saved to: output/synthetic_experiments/<f_obj.name>/results_<f_obj.name>-<dim>_<date>.svg

    Parameters
    ----------
    f_obj : object
        Objective function object (expects .name and .f.d attributes used in filename).
    kappas : sequence-like of length 4
        kappa values for the four models in order:
         [ExactGP, AdditiveGP, SobolGP, MHGP]
    n_init, n_iter, n_reps : ints
        BO parameters (passed to run_bo/run_sobolbo)
    ci : int (default=95)
        Confidence interval percentage for regret shading (only 95 supported well -> z=1.96).
    """

    # Map models -> (class, color, label, kappa)
    model_specs = [
        (ExactGP, 'red',  'ExactGP',    kappas[0]),
        (AdditiveGP, 'blue', 'AdditiveGP',   kappas[1]),
        (SobolGP, 'green', 'SobolGP',             kappas[2]),
        #(MHGP, 'orange', 'MHGP',                  kappas[3])
    ]

    # Pre-allocate results containers
    # Structure: raw_results[label] = list of result dicts
    raw_results = {spec[2]: [] for spec in model_specs}
    
    # 1. Prepare Jobs
    jobs = []
    num_workers = len(devices)
    job_idx = 0
    
    for model_cls, color, label, kappa in model_specs:
        for rep in range(n_reps):
            device = devices[job_idx % num_workers]
            jobs.append({
                'model_label': label,
                'model_cls': model_cls,
                'kappa': kappa,
                'rep': rep,
                'device': device,
                'f_obj': f_obj,
                'n_init': n_init,
                'n_iter': n_iter,
            })
            job_idx += 1

    print(f"\nStarting Optimization Metrics with {len(jobs)} total jobs on {num_workers} workers.")

    # 2. Execute Parallel Jobs
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_optm, **job) for job in jobs]
        
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                label, data = res
                raw_results[label].append(data)

    # 3. Aggregate and Plot (The rest is mostly your original logic)
    print("\nProcessing results and generating plots...")
    metrics = {}
    regrets_results = {}

    for model_cls, color, label, kappa in model_specs:
        data_list = raw_results[label]
        if not data_list:
            print(f"Warning: No results for {label}")
            continue
            
        # Stack results
        all_regrets = np.array([d['regrets'] for d in data_list])
        all_explore = np.array([d['exploration'] for d in data_list])
        all_exploit = np.array([d['exploitation'] for d in data_list])
        all_r2 = np.array([d['r2'] for d in data_list])

        # Averages
        mean_explore = np.mean(all_explore, axis=0)
        mean_exploit = np.mean(all_exploit, axis=0)
        mean_regrets = np.mean(all_regrets, axis=0)
        mean_r2 = np.mean(all_r2, axis=0)
        std_regrets = np.std(all_regrets, axis=0)

        metrics[label] = {
            'explore': mean_explore,
            'exploit': mean_exploit,
            'R2': mean_r2,
            'color': color,
            'kappa': kappa,
        }

        # Confidence Interval
        ci_scale = 1.96 if ci == 95 else 1.0
        # safety check for single rep
        denom = np.sqrt(len(data_list)) if len(data_list) > 0 else 1.0
        ci_regrets = ci_scale * std_regrets / denom
        
        regrets_results[label] = {
            'mean': mean_regrets, 
            'ci': ci_regrets, 
            'color': color, 
            'kappa': kappa
        }


    
    # ----------------------
    # Figure 1: Performance (R², Exploration, Exploitation)
    # ----------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # For each model, determine a T (shortest available among metrics) and plot
    for label, vals in metrics.items():
        color = vals['color']
        explore = vals['explore']
        exploit = vals['exploit']
        r2 = vals['R2']
        kappa = vals['kappa']
        T = exploit.shape[0]
        it = np.arange(1, T + 1)
        ax1.plot(it, explore[:T], linestyle='-', color=color, label=f'{label} kappa {kappa}')
        ax1.plot(it, exploit[:T], linestyle='-.', color=color)
        ax1.plot(it, r2[:T], linestyle=':', color=color)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Metric')
    ax1.set_title(f'Average Performance over {n_reps} runs | {f_obj.d}-{f_obj.name}')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize='small')

    # Save performance figure
    output_dir = os.path.join('output', 'synthetic_experiments', f_obj.name)
    os.makedirs(output_dir, exist_ok=True)
    dim = f_obj.d
    perf_filename = f'explr-explt_{dim}d-{f_obj.name}_budget{n_iter}.svg'
    perf_path = os.path.join(output_dir, perf_filename)
    plt.savefig(perf_path, format='svg')
    plt.close(fig1)
    print(f"Saved performance plot to {perf_path}")

    # ----------------------
    # Figure 2: Regrets (log scale) with CI shading
    # ----------------------
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    iterations = np.arange(1, n_iter -n_init + 1)
    eps = 1e-12  # small positive to avoid log(0)
    global_min, global_max = float('inf'), 0.0

    for label, vals in regrets_results.items():
        mean_reg = vals['mean']
        ci_reg = vals['ci']
        color = vals['color']
        kappa = vals['kappa']

        # lower/upper bounds (clip to eps so log plotting works)
        lower = np.maximum(mean_reg - ci_reg, eps)
        upper = np.maximum(mean_reg + ci_reg, eps)
        mean_plot = np.maximum(mean_reg, eps)

        # update global bounds so we can set y-limits to include all curves
        global_min = min(global_min, lower.min())
        global_max = max(global_max, upper.max())

        ax2.plot(it, mean_reg, color=color, label=f'{label} kappa={kappa}')
        ax2.fill_between(it, mean_reg - ci_regrets, mean_reg + ci_regrets, color=color, alpha=0.2)


    # Snap y-limits to full decades (powers of ten) so ticks are exactly 10^k
    log10_min = np.floor(np.log10(global_min))
    log10_max = np.ceil(np.log10(global_max))
    log10_max = max(log10_max, 10)
    ymin = 10.0 ** log10_min
    ymax = 10.0 ** log10_max

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Regret (log scale)')
    ax2.set_title(f'Mean Log-Regret across {n_reps} runs | {f_obj.d}-{f_obj.name}')
    ax2.set_yscale("log")
    ax2.autoscale(enable=True, axis='y', tight=None) 
    ax2.grid(True)
    ax2.legend(loc='upper right', fontsize='small')

    regrets_filename = f'regrets_{dim}d-{f_obj.name}_budget{n_iter}.svg'
    regrets_path = os.path.join(output_dir, regrets_filename)
    plt.savefig(regrets_path, format='svg')
    plt.close(fig2)
    print(f"Saved regret plot to {regrets_path}")

    # Return numeric results and paths
    return {
        'metrics': metrics,
        'regrets': regrets_results,
        'perf_path': perf_path,
        'regrets_path': regrets_path
    }

def partition_reconstruction(f_obj,  model_cls, n_init=1, n_iter=200, n_reps= 10, n_sobol=10, kappa=1.0, threshold = 0.05, save=False, verbose=False):

    """
    Compute and plot how well partition_updates reconstruct the true Sobol interaction graph.

    Returns:
        cc_list: list of CC values (one per sobol update)
        cs_list: list of CS values (one per sobol update)
        update_iters: list of BO iteration numbers corresponding to each sobol update
    """
    # 1) Get True Sobol Interactions
    sobols = sobol_sensitivity(f_obj, n_samples=100000)  # shape (d, d) expected
    sobols = np.nan_to_num(np.asarray(sobols, dtype=float))
    sobols_sym = 0.5 * (sobols + sobols.T)
    np.fill_diagonal(sobols_sym, 0.0)
    d = f_obj.d

    # 3) Build G_true adjacency matrix
    non_additive = sobols_sym > threshold  
    additive = ~non_additive.copy()
    np.fill_diagonal(non_additive, False)
    np.fill_diagonal(additive, False)

    triu_idx = np.triu_indices(d, k=1)
    true_non_additive_count = int(np.sum(non_additive[triu_idx]))
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

    surrogate_sobols = []

    plt.figure(figsize=(9, 5))
    update_iters = list(range(n_iter))
    x = np.array(update_iters)
    color = 'green' if model_cls.__name__ == 'SobolGP' else 'orange'
    cc_label_done = False
    cs_label_done = False


    for rep in range(n_reps):

        results = run_partitionbo(f_obj,
                            model_cls=model_cls,
                            n_init=n_init,
                            n_iter=n_iter,
                            n_sobol=n_sobol,
                            kappa=kappa,
                            save=False,
                            verbose=verbose)
        
        partitions = results['partition_updates']
        surrogate_sobols.append(results['sobol_interactions'])

        # 4) Plot CC and CS trace for this repetition
        cc_list = []
        cs_list = []

        prev_group_of = None
        partition_changed_flags = []

        for t, P in enumerate(partitions):

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
            cc = non_additive_overlap_count / true_non_additive_count if true_non_additive_count > 0 else 0.0
            nonedge_pred = ~G_exp
            additive_overlap = np.logical_and(additive, nonedge_pred)
            np.fill_diagonal(additive_overlap, False)
            additive_overlap_count = int(np.sum(additive_overlap[triu_idx]))
            cs = additive_overlap_count / true_additive_count if true_additive_count > 0 else 0.0
            cc_list.append(cc)
            cs_list.append(cs)

        cc_list = [0.0] * n_init + cc_list
        cs_list = [0.0] * n_init + cs_list
        partition_changed_flags = [False] * n_init + partition_changed_flags

        # Plot CC (match of true edges)
        cc_label = 'CC (true-edge overlap)' if not cc_label_done else '_nolegend_'
        plt.plot(x, cc_list, label=cc_label, linestyle='-', color=color, alpha=0.3)
        change_x = x[np.array(partition_changed_flags, dtype=bool)]
        change_cc = np.array(cc_list)[np.array(partition_changed_flags, dtype=bool)]
        if change_x.size > 0:
            plt.plot(change_x, change_cc, linestyle='None', marker='o', color=color, alpha=0.3)
        cc_label_done = True

        # Plot CS (match of true non-edges)
        cs_label = 'CS (true-nonedge overlap)' if not cs_label_done else '_nolegend_'
        plt.plot(x, cs_list, label=cs_label, linestyle='--', color='dimgrey', alpha=0.3)
        change_cs = np.array(cs_list)[np.array(partition_changed_flags, dtype=bool)]
        if change_x.size > 0:
            plt.plot(change_x, change_cs, linestyle='None', marker='s', color='dimgrey', alpha=0.3)
        cs_label_done = True

    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction score')
    plt.ylim(0.0, 1.1)
    plt.title(f'Partition reconstruction for {f_obj.d}d-{f_obj.name} over {n_reps} reps | {model_cls.__name__} (kappa={kappa}, Sobol threshold = {threshold})')
    plt.grid(True)
    plt.legend()
    
    output_dir = os.path.join('output', 'synthetic_experiments', f_obj.name)
    os.makedirs(output_dir, exist_ok=True)
    fname = f'partition_recon_{model_cls.__name__}_{f_obj.d}d-{f_obj.name}_kappa{kappa}.png'
    outpath = os.path.join(output_dir, fname)
    plt.savefig(outpath, dpi=200)
    if verbose:
        print(f"Saved partition reconstruction plot to {outpath}")

   

    ### Figure 2: Sobol interaction traces
    pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    n_pairs = len(pairs)
    nrows = int(n_pairs)

    fig2, axes = plt.subplots(nrows, 1, figsize=(8, 3*nrows + 1), squeeze=False)
    axes_flat = axes.flatten()
    color = 'green' if model_cls.__name__ == 'SobolGP' else 'orange'
    labels = ['True interaction', 'Additivity threshold', 'Predicted interaction']

    x_full = np.arange(1, n_iter + 1)
    for idx, (i, j) in enumerate(pairs):
        ax = axes_flat[idx]

        # Plot true as horizontal black line
        true_sobol = float(sobols_sym[i,j])
        true_label = ax.plot(x_full, [true_sobol]*x_full.shape[0], color='black', linestyle='--', linewidth=1.2)
        threshold_label = ax.plot(x_full, [threshold]*x_full.shape[0], color='red', linestyle='--', linewidth=1.2)

        # Estimated sobol interaction curve
        surrogate_mean = []
        for surrogate_sobol in surrogate_sobols:

            sobol_trace = [1.0]*n_init
            for t in range(n_iter-n_init):
                surrogate_estimation = surrogate_sobol[t]
                sobol_trace.append(float(surrogate_estimation[i,j]))
            sobol_trace = np.asarray(sobol_trace, dtype=float)
            surrogate_mean.append(sobol_trace)
            ax.plot(x_full, sobol_trace, color=color, linestyle='-', alpha=0.1)

        surrogate_mean = np.asarray(surrogate_mean)
        surrogate_mean = np.mean(surrogate_mean, axis=0)
        prediction_label = ax.plot(x_full, surrogate_mean, color=color, linestyle='-')

        ax.set_ylim(-0.05, 0.5)
        ax.set_xlim(1, n_iter)
        ax.set_title(f'x{i}-x{j} interaction')
        ax.grid(True, linestyle='--', linewidth=0.4)

    # wrap long suptitle into multiple lines so it doesn't overflow
    raw_title = f'Sobol reconstruction for {f_obj.d}-{f_obj.name} | {model_cls.__name__} (kappa={kappa}, Sobol threshold={threshold}'
    wrapped_title = textwrap.fill(raw_title, width=80)

    fig2.suptitle(wrapped_title, fontsize=14)
    fig2.legend([true_label, threshold_label, prediction_label], labels=labels, loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()

    sobol_fname = f'sobol_recon_{model_cls.__name__}_{f_obj.d}d-{f_obj.name}_kappa{kappa}.svg'
    outpath = os.path.join(output_dir, sobol_fname)
    fig2.savefig(outpath)
    plt.close(fig2)
   
### Parser handling

def _parse_list_of_floats(text):
    if text is None:
        return None
    try:
        return [float(x) for x in text.split(',') if len(x.strip())]
    except Exception:
        raise argparse.ArgumentTypeError('Expected comma-separated list of numbers (e.g. 0.5,1,3)')

def _parse_list_of_strings(arg):
    return arg.split(',')

def _parse_list_of_ints(text):
    if text is None:
        return None
    try:
        return [int(x) for x in text.split(',') if len(x.strip())]
    except Exception:
        raise argparse.ArgumentTypeError('Expected comma-separated list of ints (e.g. 1,2,3)')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Refactored partitionGPBO runner with CLI options')

    # Function / problem selection
    parser.add_argument('--f_ob', type=str, default='twoblobs', help='Name of synthetic function (e.g. ackley_correlated)')
    parser.add_argument('--dim', type=int, default=2, help='Dimension d for the synthetic function')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level for SyntheticTestFun')
    parser.add_argument('--negate', choices=['auto','true','false'], default='auto', help='Negate objective? if auto will use default mapping in script')

    # Method selection
    parser.add_argument('--method', type=str, default='run_bo', help='Which method to run: run_bo, run_partitionbo, kappa_search, optimization_metrics, partition_reconstruction')
    parser.add_argument('--model_cls', type=str, default='ExactGP', help='Model class name (ExactGP, AdditiveGP, SobolGP, MHGP)')
    parser.add_argument('--bo_method', type=str, default='run_bo', help='BO method used by higher-level routines (run_bo or run_partitionbo)')

    # Method-specific params
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--n_sobol', type=int, default=20)
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--kappa_list', type=_parse_list_of_floats, default=None, help='Comma-separated kappas for kappa_search')
    parser.add_argument('--kappas', type=_parse_list_of_floats, default=None, help='Comma-separated kappas for optimization_metrics (3 values expected)')

    #  Device Flags ---
    parser.add_argument('--device', type=str, default='cpu', help='Device for single runs (e.g. cuda:0)')
    parser.add_argument('--devices', type=_parse_list_of_strings, default=None, 
                        help='Comma-separated list of devices for parallel search (e.g. cuda:0,cuda:1)')

    # Misc
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--list_models', action='store_true')
    parser.add_argument('--list_methods', action='store_true')

    args = parser.parse_args(argv)

    # Allowed mappings (whitelist)
    model_map = {
        'ExactGP': ExactGP,
        'AdditiveGP': AdditiveGP,
        'SobolGP': SobolGP,
        'MHGP': MHGP,
    }

    method_map = {
        'run_bo': run_bo,
        'run_partitionbo': run_partitionbo,
        'kappa_search': kappa_search,
        'optimization_metrics': optimization_metrics,
        'partition_reconstruction': partition_reconstruction,
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

    if args.bo_method not in ['run_bo', 'run_partitionbo']:
        raise ValueError("bo_method must be 'run_bo' or 'run_partitionbo'")

    model_cls = model_map[args.model_cls]
    bo_method = run_partitionbo if args.model_cls in ['MHGP', 'SobolGP'] else run_bo
    n_init = args.dim*3

    # Construct SyntheticTestFun object
    if args.dim is None:
        raise ValueError('You must specify --dim for the function')

    # Determine negate default if user passed auto
    if args.negate == 'auto':
        negate_default_names = ['twoblobs', 'dblobs', 'multprod', 'cyclical-fun']
        negate = False if args.f_ob in negate_default_names else True
    else:
        negate = True if args.negate == 'true' else False

    f_obj = SyntheticTestFun(name=args.f_ob, d=args.dim, noise=args.noise, negate=negate)

    device_list = args.devices if args.devices else [args.device]

    # Call the requested method with the right parameter signatures
    print(f"Running method={args.method} model_cls={args.model_cls} on {args.f_ob} (d={args.dim})")

    # dispatch
    try:
        if args.method == 'run_bo':
            result = run_bo(f_obj, model_cls, n_iter=args.n_iter, kappa=args.kappa, 
                            save=args.save, verbose=args.verbose, device=args.device)
        
        elif args.method == 'run_partitionbo':
            result = run_partitionbo(f_obj, model_cls, n_iter=args.n_iter, n_sobol=args.n_sobol, kappa=args.kappa, 
                            save=args.save, verbose=args.verbose, device=args.device)

        elif args.method == 'kappa_search':
            # Provide a default kappa list if none given
            k_list = args.kappa_list or [0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 15.0]
            result = kappa_search(f_obj, k_list, model_cls=model_cls, n_iter=args.n_iter, n_reps=args.n_reps, 
                                  bo_method=bo_method, devices=device_list)

        elif args.method == 'optimization_metrics':
            if args.kappas is None: raise ValueError('--kappas must be provided for optimization_metrics (comma-separated 4 values)')
            kappas = args.kappas
            result = optimization_metrics(f_obj, kappas, n_iter=args.n_iter, n_reps=args.n_reps, devices=device_list)
        elif args.method == 'partition_reconstruction':
            result = partition_reconstruction(f_obj, model_cls, n_iter=args.n_iter, n_reps= args.n_reps, n_sobol=args.n_sobol, kappa=args.kappa, save=args.save, verbose=args.verbose)
        else:
            raise ValueError('Unsupported method')

        print('Completed. Result summary:')

    except Exception as e:
        print('ERROR during execution:', e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':

    main()
    #f_obj = SyntheticTestFun('michalewicz', 2, False, True).to('cuda:1')
    #results = run_partitionbo(f_obj, SobolGP, n_iter=50, kappa=3.0, device='cuda:1')
    #print('done')