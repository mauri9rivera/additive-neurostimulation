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

from concurrent.futures import ThreadPoolExecutor
import warnings
import traceback

### MODULES HANDLING ###
# If the package structure differs you may need to adjust PYTHONPATH or the imports below
from models.gaussians import AdditiveKernelGP, BaseGP, WrappedModel, ExactGPModel
from utils.synthetic_datasets import *

warnings.filterwarnings("ignore", category=FutureWarning, module="SALib.util")


### GLOBAL VARIABLES ###
DEVICE = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- GP classes ----------

class MHGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, partition = None, sobol=None, epsilon=5e-2):

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean() 
        self.partition = partition if partition is not None else [[i] for i in range(train_x.shape[-1])]
        self.sobol = sobol
        self.name = 'MetropolisHastingsGP'
        self.n_dims = train_x.shape[-1]
        self.epsilon = 0.03 - 0.02*(np.min(1.0, (self.n_dims**2)/40))
        self.split_bias = 0.5

        #build covar_module based on partition
        self._build_covar()

    def _build_covar(self):

        kernels = []
        for group in self.partition:

            ard_dims = len(group)

            base_kernel = gpytorch.kernels.MaternKernel(
                nu = 2.5,
                ard_num_dims = ard_dims,
                active_dims = group,
            )

            scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel)

            kernels.append(scaled_kernel)

        self.covar_module = gpytorch.kernels.AdditiveKernel(*kernels)

    def partition_to_key(self, partition):
        # Convert to a hashable, sorted tuple of tuples
        return [list(map(int, grp)) for grp in partition if len(grp) > 0] #tuple(sorted(tuple(sorted(sub)) for sub in partition))

    def update_partition(self, new_partition):

        # normalize indices to ints
        new_partition = self.partition_to_key(new_partition)
        # avoid unnecessary rebuilds:
        if new_partition == self.partition:
            return
        self.partition = new_partition
        self._build_covar()
    
    def sobol_partitioning(self, interactions):
        
        old_partition = self.partition
        has_candidate = False
        attempts = 0

        while not has_candidate and attempts < 50:

            new_partition = copy.deepcopy(self.partition)
            
            # Split strategy
            if random.random() < self.split_bias:

                robbed_subset = list(new_partition[random.randint(0, len(new_partition) - 1)])

                if len(robbed_subset) > 1:

                    victim = np.random.choice(robbed_subset) #, random.randint(1, len(splitted_subset) - 1))
                    robbed_subset.remove(victim)
                    new_partition.append([victim.astype(int)]) 
                
                    if self.are_additive(victim, robbed_subset, interactions):

                        has_candidate = True
                        return new_partition

            # Merge strategy
            else:
                
                if len(new_partition) > 1:

                    robber_idx, robbed_idx = random.sample(range(len(new_partition)), 2)

                    robbed_subset = new_partition[robbed_idx]
                    robber_subset = new_partition[robber_idx]

                    new_partition.remove(robbed_subset)
                    new_partition.remove(robber_subset)
                    new_partition.append(robbed_subset + robber_subset)

                    all_additive = True
                    for victim in robbed_subset:
                            additive = self.are_additive(victim, robber_subset, interactions)
                            if not additive:
                                all_additive = False
                                break
                    
                    has_candidate = all_additive
                    if has_candidate:
                        return new_partition
 
            attempts +=1

        print(f'Number of attempts exceedded')
        return old_partition

    def are_additive(self, individual, set, interactions):

        for evaluator in set:

            i = min(evaluator, individual)
            j = max(evaluator, individual)

            if interactions[j][i] > self.epsilon:
                return False
        
        return True
        
    def calculate_acceptance(self, proposed_model):

        self.eval()
        self.likelihood.eval()
        proposed_model.eval()
        proposed_model.likelihood.eval()

        mll_curr = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll_proposed = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, proposed_model)
        train_inputs = tuple(t.to(DEVICE) for t in self.train_inputs)

        curr_evidence = -mll_curr(self(*train_inputs), self.train_targets)
        proposed_evidence = -mll_proposed(proposed_model(proposed_model.train_inputs[0]), proposed_model.train_targets)

        acceptance = min(1, (proposed_evidence / curr_evidence))

        return acceptance
    
    def reconfigure_space(self, surrogate, surrogate_likelihood, device=DEVICE):

        # update sobol interactions from surrogate model training
        self.sobol.update_interactions(surrogate.train_inputs[0], surrogate.train_targets, surrogate, surrogate_likelihood)
        interactions = self.sobol.interactions
        new_partition = self.sobol_partitioning(interactions)
        proposed_model = MHGP(self.train_inputs[0], self.train_targets, gpytorch.likelihoods.GaussianLikelihood(), 
                              partition=new_partition)
        proposed_model.to(device)
        proposed_model.likelihood.to(device)
        proposed_model, proposed_model.likelihood, _ = optimize(proposed_model, proposed_model.train_inputs[0], proposed_model.train_targets, n_iter=20, lr=0.01)
        
        acceptance_ratio = self.calculate_acceptance(proposed_model)

        if acceptance_ratio >= 0.9:
            print(f'Update for model accepted! Genetics propose to partition {new_partition} from {self.partition} with acceptance {acceptance_ratio}')
            self.update_partition(new_partition)
            print(f'new partition: {self.partition}')
            #partition_key, self.history()
        
        return interactions
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SobolGP(gpytorch.models.ExactGP):

    """
    Exact GP that composes an additive kernel according to `partition`.
    - partition: list of lists of integer dimension indices, e.g. [[0,2],[1,3]]
    - sobol: an associated Sobol object (optional)
    - epsilon: additivity threshold (kept as attribute)
    """
    def __init__(self, train_x, train_y, likelihood, partition=None, sobol=None, epsilon=5e-2):
        super(SobolGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.partition = partition if partition is not None else [[i] for i in range(train_x.shape[-1])]
        self.sobol = sobol
        self.n_dims = train_x.shape[-1]
        self.epsilon = 0.03 - 0.02*(np.min(1.0, (self.n_dims**2)/40))
        self.name = 'SobolGP'
        # build covar_module based on partition
        self._build_covar()

    def _build_covar(self):

        kernels = []
        for group in self.partition:

            ard_dims = len(group)

            base_kernel = gpytorch.kernels.MaternKernel(
                nu = 2.5,
                ard_num_dims = ard_dims,
                active_dims = group,
            )

            scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel)

            kernels.append(scaled_kernel)

        self.covar_module = gpytorch.kernels.AdditiveKernel(*kernels)

    def update_partition(self, new_partition):

        # normalize indices to ints
        new_partition = [list(map(int, grp)) for grp in new_partition if len(grp) > 0]
        # avoid unnecessary rebuilds:
        if new_partition == self.partition:
            return
        self.partition = new_partition
        self._build_covar()

    def reconfigure_space(self, surrogate, surrogate_likelihood):
    
        # update Sobol interactions based on new surrogate train data
        self.sobol.update_interactions(surrogate.train_inputs[0], surrogate.train_targets, surrogate, surrogate_likelihood)
        new_partition = self.sobol.update_partition()
        self.update_partition(new_partition)

        return self.sobol.interactions

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Sobol:
    """
    Uses SALib to compute variance-based Sobol indices (including second-order),
    and produce a partition based on thresholds of interaction strength.

    Attributes:
      epsilon : float threshold on |S2_ij| above which we consider (i,j) interacting
      problem : dict for SALib problem definition (names, bounds, etc.)
        Initialized when first compute_interactions is called.

    Methods:
      compute_interactions(train_x, train_y) -> interactions matrix (d x d numpy)
      update_partition(interactions) -> partition (list of list of dims)
    """

    def __init__(self, f_obj, epsilon=5e-2, n_sobol_samples=None):
        """
        epsilon: threshold for second-order interactions
        n_sobol_samples: number of base samples N used by SALib's Saltelli sampler.
            If None, some default (e.g. 1024) will be chosen.
        """
        self.epsilon = float(epsilon)
        self.n_sobol_samples = n_sobol_samples
        self.problem = self._build_problem(f_obj)  # to be set up when know input bounds and d
        self.interactions = None

    def _build_problem(self, f_obj):
        """
        Build SALib 'problem' dict from data array x_np shape (n, d).
        Uses per-dimension observed min/max as bounds (assumes uniform marginals).
        """
        d = f_obj.d
        lb = f_obj.lower_bounds
        ub = f_obj.upper_bounds
        problem = {
            'num_vars': int(d),
            'names': np.array([f"x{i}" for i in range(d)]),
            'bounds': np.stack([lb, ub], axis=1).tolist()
        }

        return problem

    def update_interactions(self, train_x, train_y, simulator, likelihood):
        """
        train_x: Tensor (n, d) or array
        train_y: Tensor or array (n,) or (n,1)
        simulator: a trained ExactGP model
        likelihood: corresponding GPyTorch likelihood

        Returns:
          interactions: numpy array (d, d), symmetric, with entries S2[i,j]
        """

        
        # train surrogate model
        simulator, likelihood, _ = optimize(simulator, train_x, train_y,)

        try:

            # Convert to numpy
            if isinstance(train_x, torch.Tensor):
                x = train_x.detach().cpu().numpy()
            else:
                x = np.asarray(train_x)

            if isinstance(train_y, torch.Tensor):
                y = train_y.detach().cpu().numpy()
            else:
                y = np.asarray(train_y)

            # Flatten y
            y = y.reshape(-1)

            n, d = x.shape

            # Determine how many samples to use
            N = self.n_sobol_samples or (n // (2 * (d + 2)))
            N = max(N, 128)

            # Generate Saltelli samples
            param_values = sobol_sampler.sample(self.problem, N, calc_second_order=True, skip_values=N*2)

            # Evaluate simulator at param_values
            param_values = np.asarray(param_values, dtype=np.float32)
            m_total = param_values.shape[0]
            y_s_list = []
            batch_size = 2048

            simulator.eval(), likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for start in range(0, m_total, batch_size):
                    end = min(start + batch_size, m_total)
                    batch_np = param_values[start:end]
                    batch_t = torch.from_numpy(batch_np).float()
                    post = likelihood(simulator(batch_t))
                    y_s_list.append(post.mean.cpu().numpy())

            y_s = np.concatenate(y_s_list, axis=0)

            if np.var(y_s) < 1e-12 or np.unique(y_s).size <= 1:
                # Option A: return zero interactions (no interactions found)
                d = self.problem['num_vars']
                S2_sym = np.zeros((d, d), dtype=float)
                np.fill_diagonal(S2_sym, 0.0)
                self.interactions = S2_sym
                return S2_sym

            # Analyze with SALib
            Si = salib_sobol.analyze(
                self.problem, y_s,
                calc_second_order=True, print_to_console=False
            )

            if 'S2' not in Si:
                raise RuntimeError("SALib did not return 'S2'. Ensure calc_second_order=True.")

            # Build symmetric interactions matrix
            S2 = np.asarray(Si['S2'], dtype=float)
            S2 = np.nan_to_num(S2, nan=0.0)
            S2 = np.clip(S2, a_min=0.0, a_max=None)

            # Symmetrize and zero diagonal
            #S2_sym = 0.5 * (S2 + S2.T)
            np.fill_diagonal(S2, 1.0)

            self.interactions = S2
            return S2


        except Exception as e:
            print("compute_interactions: EXCEPTION:", e)
            traceback.print_exc(file=sys.stdout)
            # re-raise so the Future contains the exception (better than returning None)
            raise

    def update_partition(self):
        """
        Partition dimensions using a greedy algorithm based on 2nd-order interactions.

        Inputs:
        - interactions: numpy array (d, d) symmetric matrix with 2nd-order Sobol indices.
                        Only the upper diagonal is needed but full symmetric is expected.
        Output:
        - partitions: list of lists, each sublist contains indices belonging to a partition.
        """
        d = self.problem['num_vars']

        # initialize stack of dimensions
        S = list(range(d))

        # P will hold the partitions
        P = []
        # seed P with first element
        P.append([S.pop()])

        while S:
            e = S.pop()
            additive = True
            modif = False

            for sub in P:
                if modif:
                    break
                for x in sub:
                    if modif:
                        break
                    if self.interactions[x, e] > self.epsilon:
                        additive = False
                        sub.append(e)
                        modif = True

            if additive:
                P.append([e])

        return P

### Runners ###

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

    #print(f"Results for {name} ({dim}D):")
    #print("First-order indices:", Si['S1'])
    #print("Second-order indices: \n", Si['S2'])
    return Si['S2']

def optimize(gp, train_x, train_y, n_iter=20, lr=0.01):
    """
    Train an ExactGP + Likelihood model.

    Args:
        gp: ExactGP model
        likelihood: gpytorch.likelihoods.GaussianLikelihood
        train_x: torch.Tensor (n, d)
        train_y: torch.Tensor (n,)
        n_iter: int, number of training iterations
        lr: float, learning rate

    Returns:
        gp (trained), likelihood (trained), mean_epoch_loss (float)
    """

    # training inputs alignmnet
    gp.set_train_data(inputs=train_x, targets=train_y, strict=False)

    gp.train()
    gp.likelihood.train()

    optimizer = torch.optim.Adam(gp.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    epoch_losses = []
    for i in range(n_iter):
        optimizer.zero_grad()
        output = gp(train_x)
        loss = -mll(output, train_y)

        if loss.dim !=0:
            loss = loss.sum()

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    mean_loss = float(np.mean(epoch_losses))
    return gp, gp.likelihood, mean_loss

def maximize_acq(kappa_val, gp_model, gp_likelihood, grid_points):
        """
        Grid-search UCB maximizer.
        Returns: new_x (1 x d tensor), ucb_value (float), idx (int)
        UCB = mean + kappa * std
        """
        gp_model.eval()
        gp_likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = gp_likelihood(gp_model(grid_points))
            mean = post.mean           # shape (n_points,)
            var = post.variance
            std = var.clamp_min(0.0).sqrt()
            ucb = mean + kappa_val * std

        best_idx = torch.argmax(ucb).item()
        best_x = grid_points[best_idx].unsqueeze(0)  # keep shape (1, d)
        return best_x, ucb[best_idx].item(), best_idx

def run_partitionbo(f_obj,  model_cls=SobolGP, n_init=1, n_iter=200, n_sobol=10, kappa=1.0, acq_method = 'grid', save=False, verbose=False):
    """
    Returns the same metrics as run_bo, plus:
      - partition_updates: list of partition structures (list of lists) when updates occurred
      - sobol_interactions: list of interaction matrices (numpy arrays) when updates occurred
    """
    # --- Configurable internal defaults (you can tweak these) ---
    sobol_workers = 1   # background threads for Sobol
    # --------------------------------------------------------

    # Initiate n_init points, bounds
    n_points = int(100 * f_obj.d**2)
    bounds = torch.from_numpy(np.stack([f_obj.lower_bounds, f_obj.upper_bounds])).float()  # shape (2, d)
    lower, upper = bounds[0], bounds[1]
    grid_x = lower + (upper - lower) * torch.rand(n_points, f_obj.d)
    with torch.no_grad():
        grid_y = f_obj.f.forward(grid_x).float().squeeze(-1)
    grid_min = grid_y.min().item()

    # Initiate training set
    true_best = grid_y.max().item() #f_obj.f.optimal_value
    train_x, train_y  = f_obj.simulate(n_init)
    best_observed = train_y.max().item()

    # initialize metrics (same as run_bo)
    r_squared, loss_curve, expl_scores, exploit_scores, regrets, train_times = [], [], [], [], [], []

    # Additional metrics
    partition_updates = []
    sobol_interactions = []

    # Initialize Sobol and executor
    executor = ThreadPoolExecutor(max_workers=sobol_workers)
    space_reconfiguration = None

    # initialize model 
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_cls(train_x, train_y, likelihood)
    sobol = Sobol(f_obj)
    model.sobol = sobol

    # Pre-train a surrogate (ExactGP) on initial data to seed the first sobol job
    surrogate_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    surrogate = ExactGPModel(train_x, train_y, surrogate_likelihood)
    interactions = model.sobol.update_interactions(train_x, train_y, surrogate, surrogate_likelihood)
    
    # main optimization loop
    for i in range(n_iter - n_init):

        # Train model
        t0 = time.time()
        model, likelihood, epoch_losses = optimize(model, train_x, train_y)
        loss_curve.append(np.mean(epoch_losses))

        # Acquisition scoring
        if acq_method == 'botorch':
            UCB = UpperConfidenceBound(model=model, beta=kappa**2, maximize=True) #now this won't work
            # Next point
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            new_x, _ = optimize_acqf(
                acq_function=UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=512)
            
        else:
            new_x, acq_val, acq_idx = maximize_acq(kappa, model, likelihood, grid_x)

        new_y = f_obj.f.forward(new_x)

        # update dataset
        train_x = torch.cat([train_x, new_x])
        train_y  = torch.cat([train_y, new_y])
        if train_y.min().item() < grid_min:
            grid_min = train_y.min().item()
        if train_y.max().item() > true_best:
            true_best = train_y.max().item()

        # Posterior predictions for R^2
        model.eval(); likelihood.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning if hasattr(gpytorch.utils, "warnings") else Warning)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                post = likelihood(model(train_x))

            y_true = train_y.squeeze(-1)
            y_pred = post.mean
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
        exploit_scores.append(exploit)
        expl_scores.append(explore)

        # Regret scores
        regret = true_best - best_observed + 1e-9
        regrets.append(regret)

        # If there is a pending Sobol job finished, fetch it and update partition
        if space_reconfiguration is not None and space_reconfiguration.done():

            try:
                interactions = space_reconfiguration.result()                
                space_reconfiguration = None

            except Exception as e:
                space_reconfiguration = None
                print(f"[Sobol] background job failed: {e}")


        model = model_cls(train_x, train_y, likelihood, model.partition)
        t1 = time.time()
        elapsed_train = t1 - t0
        train_times.append(elapsed_train)
        model.sobol = sobol 
        sobol_interactions.append(interactions.copy())
        partition_updates.append(model.partition)

        # Submit a new Sobol background job every n_sobol iterations (if none pending)
        if (i % n_sobol) == 0:
            if space_reconfiguration is None or space_reconfiguration.done():
                try:
                    surrogate_likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    surrogate = ExactGPModel(train_x, train_y, surrogate_likelihood)
                    space_reconfiguration = executor.submit(model.reconfigure_space, surrogate, surrogate_likelihood)
                    #sobol_future = executor.submit(sobol.compute_interactions, train_x, train_y, surrogate, surrogate_likelihood)
                except Exception as e:
                    space_reconfiguration = None
                    print(f"[Sobol] submit failed: {e}")

        if verbose:
            part_str = getattr(model, 'partition', None)
            print(f"Iter {i+1:02d} Loss: {loss_curve:.4f} Exploit score: {exploit:.4f}, Explore score: {explore:.4f}, R^2: {r2_score:.4f} | partition: {part_str}")
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

def run_bo(f_obj, model_cls, n_init=1, n_iter=200, kappa=1.0, acq_method = 'grid', save=False, verbose=False):

    # Initiate n_init points, bounds
    n_points = int(100 * f_obj.d**2)
    bounds = torch.from_numpy(np.stack([f_obj.lower_bounds, f_obj.upper_bounds])).float()  # shape (2, d)
    lower, upper = bounds[0], bounds[1]
    grid_x = lower + (upper - lower) * torch.rand(n_points, f_obj.d)
    with torch.no_grad():
        grid_y = f_obj.f.forward(grid_x).float().squeeze(-1) 
    grid_min = grid_y.min().item()
    
    # Initiate training set
    true_best = grid_y.max().item() #f_obj.f.optimal_value  
    train_x, train_y  = f_obj.simulate(n_init)
    best_observed = train_y.max().item()

    #initialize metrics
    r_squared, loss_curve, expl_scores, exploit_scores, regrets, train_times = [], [], [], [], [], []

    gp = WrappedModel(train_x, train_y, model_cls)

    for i in range(n_iter - n_init):

        # Train model
        t0 = time.time()
        gp.model, gp.likelihood, epoch_losses = optimize(gp.model, train_x, train_y)

        loss_curve.append(np.mean(epoch_losses))

        if acq_method == 'botorch':
            # Acquisition scoring
            UCB = UpperConfidenceBound(model=gp, beta=kappa**2, maximize=True)
            # Next point
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            new_x, _ = optimize_acqf(
                acq_function=UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=512)
            
        else:
            new_x, acq_val, acq_idx = maximize_acq(kappa, gp.model, gp.likelihood, grid_x)

        new_y = f_obj.f.forward(new_x)

        #update dataset
        train_x = torch.cat([train_x, new_x])
        train_y  = torch.cat([train_y, new_y])
        if train_y.min().item() < grid_min:
            grid_min = train_y.min().item()
        if train_y.max().item() > true_best:
            true_best = train_y.max().item()

        # Posterior predictions for R^2
        gp.model.eval(); gp.likelihood.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=GPInputWarning)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                post = gp.likelihood(gp.model(train_x))
            
            y_true = train_y.squeeze(-1)
            y_pred = post.mean
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
            r2_score = np.clip(1 - ss_res / ss_tot, 0.0, 1.0)
            #print(f"[DEBUG] it={i:02d}  ss_res={ss_res:.4f}  ss_tot={ss_tot:.4f}  raw R2={r2_score:.4f}")
            r_squared.append(r2_score.item())

        # Exploration and Exploitation scores (normalized) 
        best_observed = train_y.max().item()
        explore = (best_observed - grid_min) / (true_best - grid_min + 1e-9)
        exploit = (new_y.item() - grid_min) / (true_best - grid_min + 1e-9)
        explore = np.clip(explore, 0.0, 1.0)
        exploit = np.clip(exploit, 0.0, 1.0)
        exploit_scores.append(exploit)
        expl_scores.append(explore)

        # Regret scores
        regret = true_best - best_observed + 1e-9
        regrets.append(regret)

        
        gp = WrappedModel(train_x, train_y, model_cls)

        t1 = time.time()
        elapsed_train = t1 - t0
        train_times.append(elapsed_train)

        if verbose:
            print(f"{gp.model.name} | Iter {i+1:02d} Exploit score: {exploit:.2f}, Explore score: {explore:.3f}, R^2: {r2_score}")
            print(f'best observed: {best_observed:.2f} |  true best: {true_best:.2f} | current query: {new_y.item():.2f}')

    train_times = np.array(train_times, dtype=np.float64)
    cumulative_time = np.cumsum(train_times)

    if save:
    # Save results
        output_dir = os.path.join("output", "breaking_additivity")
        fname = f"{gp.model.name}_{datetime.date.today().isoformat()}_kappa{kappa}_{f_obj.name}.npz"
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

### Graph generation functions ###

def kappa_search(f_obj, kappa_list, model_cls=BaseGP, n_init=1, n_iter=100, n_reps=15,
                bo_method=run_bo, acq_method = 'grid'):
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

    for kappa in kappa_list:
        
        explore_list, exploit_list = [], []

        print(f"\nRunning kappa={kappa} for {model_cls.__name__} ({f_obj.name})")
        for rep in range(n_reps):

            results = bo_method(f_obj, model_cls, n_init=n_init, n_iter=n_iter, kappa=kappa, acq_method=acq_method, save=False)
            explore, exploit = results['exploration'], results['exploitation']

            explore_list.append(explore)
            exploit_list.append(exploit)
          
        stacked_explore = np.stack([explr for explr in explore_list], axis=0)
        stacked_exploit = np.stack([explt for explt in exploit_list], axis=0)

        averaged[kappa] = {'explore': stacked_explore.mean(axis=0), 'exploit': stacked_exploit.mean(axis=0)}


    # Plotting
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    n_k = len(kappa_list)
    x = np.arange(0, n_iter)

    for idx, kappa in enumerate(kappa_list):

        vals = averaged.get(kappa, None)
        color = cmap(idx % 10)

        mean_explore = vals['explore']
        mean_exploit = vals['exploit']

        # prepend n_init zeros (initialization period)
        explore_padded = np.concatenate([np.zeros(n_init), mean_explore])
        exploit_padded = np.concatenate([np.zeros(n_init), mean_exploit])
        # plot exploration as dashed line
        plt.plot(x, explore_padded, linestyle='-', marker=None, label=f'k={kappa} Explore', color=color)
        plt.plot(x, exploit_padded, linestyle='--', marker=None, color=color)


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

def optimization_metrics(f_obj, kappas, n_init=1, n_iter=100, n_reps=15, ci=95, acq_method = 'grid'):
    """
    Run BO variants for f_obj and plot:
      - Plot 1: Exploration (dashed), Exploitation (solid) averaged across n_reps
                    for ExactGPModel (red), AdditiveKernelGP (blue), SobolGP (green), MHGP (orange)
      - Plot 2: Mean regret +/- CI, plotted on a log y-scale (log-regrets)
    Results saved to: output/synthetic_experiments/<f_obj.name>/results_<f_obj.name>-<dim>_<date>.svg

    Parameters
    ----------
    f_obj : object
        Objective function object (expects .name and .f.d attributes used in filename).
    kappas : sequence-like of length 4
        kappa values for the four models in order:
         [ExactGPModel, AdditiveKernelGP, SobolGP, MHGP]
    n_init, n_iter, n_reps : ints
        BO parameters (passed to run_bo/run_sobolbo)
    ci : int (default=95)
        Confidence interval percentage for regret shading (only 95 supported well -> z=1.96).
    """

    # Map models -> (class, color, label, kappa)
    model_specs = [
        (ExactGPModel, 'red',  'ExactGPModel',    kappas[0]),
        (AdditiveKernelGP, 'blue', 'AdditiveGP',   kappas[1]),
        (SobolGP, 'green', 'SobolGP',             kappas[2]),
        (MHGP, 'orange', 'MHGP',                  kappas[3])
    ]

    # Containers for metrics
    metrics = {}
    regrets_results = {}

     # Run experiments for each model
    for model_cls, color, label, kappa in model_specs:
        all_regrets, all_explore, all_exploit, all_regrets = [], [], [], []

        for rep in range(n_reps):
            if label in ['SobolGP', 'MHGP']:
                exp_results = run_partitionbo(f_obj, model_cls, n_init=n_init, n_iter=n_iter, kappa=kappa, acq_method=acq_method)
            else:
                exp_results = run_bo(f_obj, model_cls, n_init=n_init, n_iter=n_iter, kappa=kappa, acq_method=acq_method)

            regrets, explore, exploit = exp_results['regrets'], exp_results['exploration'], exp_results['exploitation']
            all_regrets.append(regrets)
            all_explore.append(explore)
            all_exploit.append(exploit)

        # Convert to arrays and average across repetitions
        all_regrets = np.array(all_regrets)  # shape (n_reps, n_iter)
        all_explore = np.array(all_explore)
        all_exploit = np.array(all_exploit)
            
        mean_explore = np.mean(all_explore, axis=0)
        mean_exploit = np.mean(all_exploit, axis=0)
        mean_regrets = np.mean(all_regrets, axis=0)
        std_regrets = np.std(all_regrets, axis=0)

        metrics[label] = {
            'explore': mean_explore,
            'exploit': mean_exploit,
            'color': color,
            'kappa': kappa,
        }

        # Compute confidence interval (normal approx)
        ci_scale = 1.96 if ci == 95 else 1.0  # crude but works
        ci_regrets = ci_scale * std_regrets / np.sqrt(n_reps)
        regrets_results[label] = {'mean': mean_regrets, 'ci': ci_regrets, 'color': color, 'kappa': kappa,}


    # ----------------------
    # Figure 1: Performance (R², Exploration, Exploitation)
    # ----------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # For each model, determine a T (shortest available among metrics) and plot
    for label, vals in metrics.items():
        color = vals['color']
        explore = vals['explore']
        exploit = vals['exploit']
        kappa = vals['kappa']
        T = exploit.shape[0]
        it = np.arange(1, T + 1)
        if explore.size >= T:
            ax1.plot(it, explore[:T], linestyle='--', color=color, label=f'{label} kappa {kappa}')
        if exploit.size >= T:
            ax1.plot(it, exploit[:T], linestyle='-', color=color)

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
    #ax2.set_yscale('log')
    #ax2.set_ylim(ymin, ymax)
    #ax2.yaxis.set_major_locator(LogLocator(base=10.0))
    #ax2.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
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

def partition_reconstruction(f_obj,  model_cls, n_init=1, n_iter=200, n_sobol=10, kappa=1.0, threshold = 0.05, acq_method='grid', save=False, verbose=False):

    """
    Compute and plot how well partition_updates reconstruct the true Sobol interaction graph.

    Returns:
        cc_list: list of CC values (one per sobol update)
        cs_list: list of CS values (one per sobol update)
        update_iters: list of BO iteration numbers corresponding to each sobol update
    """
    # 1) Get true Sobol second-order matrix using provided helper
    sobols = sobol_sensitivity(f_obj, n_samples=100000)  # shape (d, d) expected
    sobols = np.nan_to_num(np.asarray(sobols, dtype=float))

    # make sure sobols is symmetric and zero-diagonal
    # if SALib returned upper triangular, symmetrize
    if sobols.shape[0] != sobols.shape[1]:
        raise ValueError("sobol_sensitivity returned non-square matrix")
    sobols_sym = 0.5 * (sobols + sobols.T)
    np.fill_diagonal(sobols_sym, 0.0)

    d = f_obj.d

    # 2) Run the Sobol-guided BO experiment
    results = run_partitionbo(f_obj,
                         model_cls=model_cls,
                         n_init=n_init,
                         n_iter=n_iter,
                         n_sobol=n_sobol,
                         kappa=kappa,
                         acq_method=acq_method,
                         save=False,
                         verbose=verbose)

    partitions_all = results.get('partition_updates', [])
    sobol_interactions_all = results.get('sobol_interactions', [])  # expected: list length ~ n_iter of (d,d) arrays

    # 3) Build G_true adjacency matrix using threshold 0.05 (upper diagonal)
    non_additive = sobols_sym > threshold  
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
    plt.title(f'Partition reconstruction for {f_obj.d}d-{f_obj.name}_kappa{kappa} | {model_cls.__name__} (Additivity threshold=0.05)')
    plt.grid(True)
    plt.legend()
    
    output_dir = os.path.join('output', 'synthetic_experiments', f_obj.name)
    os.makedirs(output_dir, exist_ok=True)
    fname = f'partition_recon_{model_cls.__name__}_{f_obj.d}d={f_obj.name}_kappa{kappa}.png'
    outpath = os.path.join(output_dir, fname)
    plt.savefig(outpath, dpi=200)
    if verbose:
        print(f"Saved partition reconstruction plot to {outpath}")

   

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
   

### Parser handling

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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Refactored partitionGPBO runner with CLI options')

    # Function / problem selection
    parser.add_argument('--f_ob', type=str, default='twoblobs', help='Name of synthetic function (e.g. ackley_correlated)')
    parser.add_argument('--dim', type=int, default=2, help='Dimension d for the synthetic function')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level for SyntheticTestFun')
    parser.add_argument('--negate', choices=['auto','true','false'], default='auto', help='Negate objective? if auto will use default mapping in script')

    # Method selection
    parser.add_argument('--method', type=str, default='run_bo', help='Which method to run: run_bo, run_partitionbo, kappa_search, optimization_metrics, partition_reconstruction')
    parser.add_argument('--model_cls', type=str, default='ExactGPModel', help='Model class name (ExactGPModel, AdditiveKernelGP, SobolGP, MHGP, BaseGP)')
    parser.add_argument('--bo_method', type=str, default='run_bo', help='BO method used by higher-level routines (run_bo or run_partitionbo)')
    parser.add_argument('--acq_method', type=str, default='grid', help='Acquisition function scoring method (botorch or grid)')

    # Method-specific params
    parser.add_argument('--n_init', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--n_sobol', type=int, default=10)
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--kappa_list', type=_parse_list_of_floats, default=None, help='Comma-separated kappas for kappa_search')
    parser.add_argument('--kappas', type=_parse_list_of_floats, default=None, help='Comma-separated kappas for optimization_metrics (3 values expected)')

    # Misc
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--list_models', action='store_true')
    parser.add_argument('--list_methods', action='store_true')

    args = parser.parse_args(argv)

    # Allowed mappings (whitelist)
    model_map = {
        'ExactGPModel': ExactGPModel,
        'AdditiveKernelGP': AdditiveKernelGP,
        'SobolGP': SobolGP,
        'MHGP': MHGP,
        'BaseGP': BaseGP,
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

    if args.acq_method not in ['grid', 'botorch']:
        raise ValueError("bo_method must be 'grid' or 'botorch'")

    model_cls = model_map[args.model_cls]
    method = method_map[args.method]
    bo_method = method_map[args.bo_method]

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

    # Call the requested method with the right parameter signatures
    print(f"Running method={args.method} model_cls={args.model_cls} on {args.f_ob} (d={args.dim})")

    # dispatch
    try:
        if args.method == 'run_bo':
            result = run_bo(f_obj, model_cls, n_init=args.n_init, n_iter=args.n_iter, kappa=args.kappa, acq_method=args.acq_method, save=args.save, verbose=args.verbose)
        elif args.method == 'run_partitionbo':
            result = run_partitionbo(f_obj, model_cls, n_init=args.n_init, n_iter=args.n_iter, n_sobol=args.n_sobol, kappa=args.kappa, acq_method=args.acq_method, save=args.save, verbose=args.verbose)
        elif args.method == 'kappa_search':
            # Provide a default kappa list if none given
            k_list = args.kappa_list or [0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 15.0]
            result = kappa_search(f_obj, k_list, model_cls=model_cls, n_init=args.n_init, n_iter=args.n_iter, n_reps=args.n_reps, bo_method=bo_method, acq_method=args.acq_method)
        elif args.method == 'optimization_metrics':
            if args.kappas is None:
                raise ValueError('--kappas must be provided for optimization_metrics (comma-separated 4 values)')
            kappas = args.kappas
            result = optimization_metrics(f_obj, kappas, n_init=args.n_init, n_iter=args.n_iter, n_reps=args.n_reps, acq_method=args.acq_method)
        elif args.method == 'partition_reconstruction':
            result = partition_reconstruction(f_obj, model_cls, n_init=args.n_init, n_iter=args.n_iter, n_sobol=args.n_sobol, kappa=args.kappa, acq_method=args.acq_method, save=args.save, verbose=args.verbose)
        else:
            raise ValueError('Unsupported method')

        print('Completed. Result summary:')
        # print a short summary depending on result shape
        if isinstance(result, dict):
            keys = list(result.keys())
            print('Keys in result:', keys)
        else:
            print(type(result))

    except Exception as e:
        print('ERROR during execution:', e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    
    #main()
    f = SyntheticTestFun('dblobs', 7, noise=False, negate=False)
    print(sobol_sensitivity(f, n_samples=100000))
    #run_partitionbo(f, MHGP)
   
      
    
