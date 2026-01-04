import numpy as np
import torch

from SALib.analyze import sobol as salib_sobol
from SALib.sample import saltelli

#from synthetic_datasets import *


def sobol_sensitivity(f_obj, n_samples=1000):
    """
    Compute 1st- and 2nd-order Sobol sensitivity indices for a given SyntheticTestFun object.

    Parameters
    ----------
    f_obj : SyntheticTestFun
        The synthetic test function wrapper.
    n_samples : int
        Base sample size for Saltelli sampling. Total model evaluations will be larger.

    Returns
    -------
    Si : dict
        Dictionary with 'S1', 'S1_conf', 'S2', 'S2_conf', 'ST', 'ST_conf'
    """
    # Define the problem for SALib
    problem = {
        'num_vars': f_obj.d,
        'names': [f"x{i+1}" for i in range(f_obj.d)],
        'bounds': list(zip(f_obj.lower_bounds, f_obj.upper_bounds))
    }

    # Generate Saltelli samples
    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

    # Evaluate the function
    X_torch = torch.tensor(param_values, dtype=torch.float32)
    Y_torch = f_obj.f.forward(X_torch).detach().numpy().flatten()

    # Sobol analysis
    Si = salib_sobol.analyze(problem, Y_torch, calc_second_order=True, print_to_console=False)
    return Si
