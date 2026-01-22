import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import textwrap
import sys
import os
import torch
import gpytorch
import datetime
import random
import copy
import argparse
import ast
import time
from gpytorch.models.exact_gp import GPInputWarning

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from SALib.analyze import sobol as salib_sobol
from SALib.sample import saltelli
from SALib.sample import sobol as sobol_sampler

from UQpy.sensitivity.PceSensitivity import PceSensitivity
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import *
from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion
from UQpy.sensitivity.SobolSensitivity import SobolSensitivity
from UQpy.distributions import Uniform
from UQpy.run_model.RunModel import RunModel

from scipy.stats import sobol_indices as scipy_sobol
from scipy.stats import qmc, uniform

import tntorch as tn
import torch.autograd.functional as F

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from concurrent.futures import ThreadPoolExecutor
import threading
import warnings
import traceback

from utils.synthetic_datasets import *
from utils.sensitivity_utils import *
from models.gaussians import *
from models.sobols import *


def test_surrogate(f_ob,  M, B, n_iter=50, opt_method = 'gp', sobol_method='scipy'):

    # Initiate n_init points, bounds
    n_points = int(100 * f_ob.d**2)
    bounds = torch.from_numpy(np.stack([f_ob.lower_bounds, f_ob.upper_bounds])).float()  # shape (2, d)
    lower, upper = bounds[0], bounds[1]
    grid_x = lower + (upper - lower) * torch.rand(n_points, f_ob.d)
    with torch.no_grad():
        grid_y = f_ob.f.forward(grid_x).float().squeeze(-1) 
    grid_min = grid_y.min().item()
    true_best = grid_y.max().item()

    sobol = Sobol(f_ob, f_ob.d*3, method=sobol_method, M=M, B=B)
    if opt_method == 'gp':
        train_x = torch.from_numpy(sobol_sampler.sample(sobol.problem, n_iter, calc_second_order=False)).float()
        train_y = f_ob.f.forward(train_x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        current_min = min(grid_min, train_y.min().item())
        current_max = max(true_best, train_y.max().item())

        for i in range(n_iter - f_ob.d*3):

            y_range = current_max - current_min
            train_y_norm = (train_y - current_min) / y_range

            # 3. Initialize and Optimize GP using NORMALIZED y
            gp = ExactGP(train_x, train_y_norm, likelihood)
            gp, likelihood, _ = optimize(gp, train_x, train_y_norm)

            new_x, acq_val, acq_idx = maximize_acq(5.0, gp, likelihood, grid_x)
            new_y = f_ob.f.forward(new_x) 

            # Update global bounds and exploration metrics
            new_y_val = new_y.item()
            if new_y_val < current_min:
                current_min = new_y_val
            if new_y_val > current_max:
                current_max = new_y_val
                
            # Keep your original tracking for the final score
            if new_y_val > true_best:
                true_best = new_y_val
            
            train_x = torch.cat([train_x, new_x])
            train_y  = torch.cat([train_y, new_y])

        y_range = current_max - current_min
        train_y_norm = (train_y - current_min) / y_range

        print(f'final exploration score: {torch.max(train_y) / true_best}\n\n')

    elif opt_method == 'sampler':

        train_x = torch.from_numpy(sobol_sampler.sample(sobol.problem, n_iter, calc_second_order=False)).float()
        train_y = f_ob.f.forward(train_x)
        current_min = min(grid_min, train_y.min().item())
        current_max = max(true_best, train_y.max().item())
        y_range = current_max - current_min
        train_y_norm = (train_y - current_min) / y_range
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp = ExactGP(train_x, train_y_norm, likelihood)
    

    print(f'predicted HD interactions: {sobol.method(train_x, train_y_norm, gp)}')
    print(f'\n\n')

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Iterable, Callable, List, Dict, Any


def threadpool_map(
    fn: Callable,
    jobs: Iterable[Dict[str, Any]],
    max_workers: int | None = None,
    verbose: bool = True,
):
    """
    Asynchronously execute fn(**job) for each job using a thread pool.

    Parameters
    ----------
    fn : callable
        Function to execute.
    jobs : iterable of dict
        Each dict contains keyword arguments for fn.
    max_workers : int or None
        Number of worker threads.
    verbose : bool
        Print progress and failures.

    Returns
    -------
    results : list
        List of (job, result) tuples in completion order.
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fn, **job): job
            for job in jobs
        }

        for future in as_completed(futures):
            job = futures[future]
            try:
                res = future.result()
                results.append((job, res))
                if verbose:
                    print(f"✓ Finished {job}")
            except Exception as e:
                if verbose:
                    print(f"✗ Failed {job}: {e}")
                results.append((job, e))

    return results

def build_jobs(
    f_ob,
    grid_M,
    grid_B,
    grid_NZ,
    methods,
):
    jobs = []
    for M, B, NZ, method in product(grid_M, grid_B, grid_NZ, methods):
        jobs.append(
            dict(
                f_ob=f_ob,
                M=M,
                B=B,
                n_iter=NZ,
                NZ=NZ,
                method=method,
            )
        )
    return jobs

def run_test_surrogate(f_ob, M, B, n_iter, method, NZ=None):
    return test_surrogate(
        f_ob,
        M=M,
        B=B,
        n_iter=n_iter,
        method=method,
    )


if __name__ == "__main__":

    f_ob = SyntheticTestFun('ishigami', 3, False, False)

    S = sobol_sensitivity(f_ob)
    real_S1 = S["S1"]
    real_ST = S["ST"]
    print(f" ----- 'True' Numerical Sobol interaction calculation -----")
    print(f'{f_ob.name} S1: {real_S1}, S_T: {real_ST} and higher order: { 1 - (real_S1 / real_ST)}\n')

    grid_NZ = [100, 150]
    grid_M = [1024, 2048]
    grid_B = [128, 256, 512]
    methods = ['scipy', 'deriv', 'tt', 'asm', 'wirthl']


    #test_surrogate(f_ob, 4096, 1024, 100, 'sampler', 'scipy')
    test_surrogate(f_ob, 4096, 1024, 30, 'gp', 'scipy')
    test_surrogate(f_ob, 4096, 1024, 30, 'gp', 'scipy')
    test_surrogate(f_ob, 4096, 1024, 30, 'gp', 'scipy')
    test_surrogate(f_ob, 4096, 1024, 30, 'gp', 'scipy')
    test_surrogate(f_ob, 4096, 1024, 30, 'gp', 'scipy')
    #test_surrogate(f_ob, 4096, 1024, 100, 'sampler', 'scipy')
    #test_surrogate(f_ob, 2048, 8, 100, 'deriv')
    #test_surrogate(f_ob, 4096, 256, 60, 'asm')

    exit(0)

    jobs = build_jobs(
        f_ob=f_ob,
        grid_M=grid_M,
        grid_B=grid_B,
        grid_NZ=grid_NZ,
        methods=methods,
    )

    results = threadpool_map(
        fn=run_test_surrogate,
        jobs=jobs,
        max_workers=4,   # tune for your CPU / I/O balance
    )

    