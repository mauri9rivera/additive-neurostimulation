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
import scipy.io
import pickle

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

# CUDA device?
DEVICE = torch.device('cpu') # torch.device("cuda:1" if torch.cuda.is_available() else "cpu") #?#


### --- neural GP classes --- ###

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

    def __init__(self, dataset_type, epsilon=5e-2, n_sobol_samples=None):
        """
        epsilon: threshold for second-order interactions
        n_sobol_samples: number of base samples N used by SALib's Saltelli sampler.
            If None, some default (e.g. 1024) will be chosen.
        """
        self.epsilon = float(epsilon)
        self.n_sobol_samples = n_sobol_samples
        self.problem = self._build_problem(dataset_type)  # to be set up when know input bounds and d
        self.interactions = None

    def _build_problem(self, dataset_type): 
        """
        Build SALib 'problem' dict from data array x_np shape (n, d).
        Uses per-dimension observed min/max as bounds (assumes uniform marginals).
        """
        if dataset_type == '5d_rat':
            d = 5
            bounds =  [[100, 400], [100, 400], [20, 200], [0, 7], [0, 3] ]
        elif dataset_type in ['spinal', 'rat']:
            bounds = [[0, 7], [0, 7]]
            d = 2
        elif dataset_type == 'nhp':
            d = 2
            bounds = [[0, 9], [0, 9]]
        else:
            d = 2
        problem = {
            'num_vars': int(d),
            'names': np.array([f"x{i}" for i in range(d)]),
            'bounds': np.asarray(bounds)
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

class AdditiveKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior, outputscale_prior):
        super().__init__(train_x, train_y, likelihood)
        self.n_dims = train_x.shape[-1]

        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims = self.n_dims, batch_shape=torch.Size([self.n_dims]),
                lengthscale_prior = lengthscale_prior  
            ),
            outputscale_prior=outputscale_prior
        )
        kernel.base_kernel.lengthscale = [1.0] * self.n_dims
        kernel.outputscale = [1.0]
        self.covar_module = kernel       

        self.mean_module = gpytorch.means.ZeroMean()
        self.likelihood = likelihood
        self.name = 'AdditiveGP'

    def forward(self, X):
        mean = self.mean_module(X)
        batched_dimensions_of_X = X.mT.unsqueeze(-1)  # Now a d x n x 1 tensor
        covar = self.covar_module(batched_dimensions_of_X).sum(dim=-3)
        return gpytorch.distributions.MultivariateNormal(mean, covar) 

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior, outputscale_prior):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.n_dims = train_x.shape[-1]
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims = self.n_dims,
                lengthscale_prior = lengthscale_prior  
            ),
            outputscale_prior=outputscale_prior
        )
        kernel.base_kernel.lengthscale = [1.0] * self.n_dims
        kernel.outputscale = [1.0]
        self.covar_module = kernel
        self.name = 'neuralMHGP'

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class neuralMHGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, lengthscale_prior, outputscale_prior, 
                 partition = None, sobol=None, epsilon=5e-2):

        super().__init__(train_x, train_y, likelihood)
        self.n_dims = train_x.shape[-1]
        self.mean_module = gpytorch.means.ZeroMean() 
        self.partition = partition if partition is not None else [[i] for i in range(train_x.shape[-1])]
        self.sobol = sobol
        self.name = 'neuralMHGP'
        self.epsilon = 0.03 - 0.02*(np.min(1.0, (self.n_dims**2)/40))
        self.split_bias = 0.5

        #build covar_module based on partition
        self.lengthscale_prior = lengthscale_prior
        self.outputscale_prior = outputscale_prior
        self._build_covar()

    def _build_covar(self):


        kernels = []
        for group in self.partition:

            ard_dims = len(group)

            base_kernel = gpytorch.kernels.MaternKernel(
                nu = 2.5,
                batch_shape=torch.Size([self.n_dims]),
                ard_num_dims = ard_dims,
                active_dims = group,
                lengthscale_prior = self.lengthscale_prior
            )
            base_kernel.lengthscale = [1.0] * self.n_dims
            scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior=self.outputscale_prior)
            scaled_kernel.outputscale = [1.0]

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
        proposed_model = neuralMHGP(self.train_inputs[0], self.train_targets, gpytorch.likelihoods.GaussianLikelihood(), 
                              partition=new_partition)
        proposed_model.to(device)
        proposed_model.likelihood.to(device)
        proposed_model, proposed_model.likelihood, _ = optimize(proposed_model, proposed_model.train_inputs[0], proposed_model.train_targets, n_iter=20, lr=0.01)
        
        acceptance_ratio = self.calculate_acceptance(proposed_model)

        if acceptance_ratio >= 0.9:
            print(f'Update for model accepted! Genetics propose to partition {new_partition} from {self.partition} with acceptance {acceptance_ratio}')
            self.update_partition(new_partition)
            #partition_key, self.history()
        
        return interactions
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class neuralSobolGP(gpytorch.models.ExactGP):

    """
    Exact GP that composes an additive kernel according to `partition`.
    - partition: list of lists of integer dimension indices, e.g. [[0,2],[1,3]]
    - sobol: an associated Sobol object (optional)
    - epsilon: additivity threshold (kept as attribute)
    """
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior, outputscale_prior, 
                 partition=None, sobol=None, epsilon=5e-2):
        super(neuralSobolGP, self).__init__(train_x, train_y, likelihood)
        self.n_dims = train_x.shape[-1]
        self.mean_module = gpytorch.means.ZeroMean()
        self.partition = partition if partition is not None else [[i] for i in range(train_x.shape[-1])]
        self.sobol = sobol
        self.epsilon = 0.03 - 0.02*(np.min(1.0, (self.n_dims**2)/40))
        self.name = 'neuralSobolGP'
        # build covar_module based on partition
        self.lengthscale_prior = lengthscale_prior
        self.outputscale_prior = outputscale_prior
        self._build_covar()

    def _build_covar(self):

        kernels = []
        for group in self.partition:

            ard_dims = len(group)

            base_kernel = gpytorch.kernels.MaternKernel(
                nu = 2.5,
                #?# batch_shape=torch.Size([self.n_dims]),
                ard_num_dims = ard_dims,
                active_dims = group,
                lengthscale_prior = self.lengthscale_prior
            )
            base_kernel.lengthscale = torch.ones(ard_dims)
            scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior=self.outputscale_prior)
            scaled_kernel.outputscale = torch.tensor(1.0)

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

### --- Runners --- ### 

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
    #print("Second-order indices:", Si['S2'])
    return Si['S2']

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

        if loss.dim() !=0:
            loss = loss.sum()

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    mean_loss = float(np.mean(epoch_losses))
    return gp, gp.likelihood, mean_loss

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
        options['n_Chan']=96
        options['n_emgs'] = 8
        options['n_dims'] = 2
    elif dataset_type == 'rat':
        options['noise_min']=0.05
        options['kappa']=3.8
        options['rho_high']=8
        options['rho_low']=0.001
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.055
        options['n_subjects']=6
        options['n_Chan']=32
        options['n_emgs'] = 8
        options['n_dims'] = 2
    elif dataset_type == '5d_rat':
        options['noise_min']=0.05
        options['kappa']=3.8
        options['rho_high']=8
        options['rho_low']=0.001
        options['nrnd']=1 #has to be >= 1
        options['noise_max']=0.055
        options['n_subjects']=6
        options['n_Chan']= 100
        options['n_emgs'] = 1
        options['n_dims'] = 5
    
    options['device'] = DEVICE
    options['n_reps'] = 20
    options['n_rnd'] = 1

    return options

def load_matlab_data(dataset_type, m_i):

    path_to_dataset = f'./datasets/{dataset_type}'

    if dataset_type == '5d_rat':

        emg_map = {
            0: np.array(['extensor carpi radialis']),
            1: np.array(['flexor carpi ulnaris']),
            2: np.array(['triceps']),
            3: np.array(['biceps']),
            4: np.array(['extensor carpi radialis']),
            5: np.array(['flexor carpi ulnaris']),

        }
        match m_i:
            case 0 | 1 | 2 | 3:

                data = scipy.io.loadmat(f'{path_to_dataset}\\rData03_230724_4x4x3x32x8_ar.mat')['Data']
                resp =  data[0][0][0]
                param = data[0][0][1]
                ch2xy = param[:, [0,1,2,5,6]]
                peak_resp = torch.from_numpy(resp).float().to(DEVICE)
                ch2xy = torch.from_numpy(ch2xy).float().to(DEVICE)
                
                subjet = {
                    'emgs': emg_map[m_i],
                    'nChan': 32,
                    'sorted_respMean': peak_resp,
                    'ch2xy': ch2xy,
                    'dim_sizes': np.array([8, 4, 3, 4, 4]),
                    'DimSearchSpace' : np.prod([8, 4, 3, 4, 4])
                }
            case 4 | 5:
                
                data = scipy.io.loadmat(f'{path_to_dataset}\\5d_step4.mat')
                resp = data['emg_response']
                param = data['stim_combinations']
                ch2xy = torch.from_numpy(param[:, [0,1,2,5,6]])
                peak_resp = torch.from_numpy(resp[:, :, :, 0]).float().to(DEVICE)


                subjet = {
                    'emgs': emg_map[m_i],
                    'nChan': 32,
                    'sorted_respMean': peak_resp,
                    'ch2xy': ch2xy,
                    'dim_sizes': np.array([8, 4, 4, 4, 4]),
                    'DimSearchSpace' : np.prod([8, 4, 4, 4, 4])
                }
                
        return subjet
    elif dataset_type=='nhp': # nhp dataset has 4 subjects
        if m_i==0:
            Cebus1_M1_190221 = scipy.io.loadmat(path_to_dataset+'/Cebus1_M1_190221.mat')
            Cebus1_M1_190221= {'emgs': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][0][0],
           'nChan': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][2][0][0],
           'sorted_isvalid': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][8],
           'sorted_resp': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][9],
           'sorted_respMean': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][10],
           'ch2xy': Cebus1_M1_190221['Cebus1_M1_190221'][0][0][16],
           'DimSearchSpace': 96},
            SET=Cebus1_M1_190221[0]
        if m_i==1:
            Cebus2_M1_200123 = scipy.io.loadmat(path_to_dataset+'/Cebus2_M1_200123.mat')  
            Cebus2_M1_200123= {'emgs': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][0][0],
           'nChan': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][2][0][0],
           'sorted_isvalid': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][8],
           'sorted_resp': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][9],
           'sorted_respMean': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][10],
           'ch2xy': Cebus2_M1_200123['Cebus2_M1_200123'][0][0][16],
           'DimSearchSpace': 96}
            SET=Cebus2_M1_200123
        if m_i==2:    
            Macaque1_M1_181212 = scipy.io.loadmat(path_to_dataset+'/Macaque1_M1_181212.mat')
            Macaque1_M1_181212= {'emgs': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][0][0],
           'nChan': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][2][0][0],
           'sorted_isvalid': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][8],
           'sorted_resp': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][9],              
           'sorted_respMean': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][15],
           'ch2xy': Macaque1_M1_181212['Macaque1_M1_181212'][0][0][14],
           'DimSearchSpace': 96}            
            SET=Macaque1_M1_181212
        if m_i==3:    
            Macaque2_M1_190527 = scipy.io.loadmat(path_to_dataset+'/Macaque2_M1_190527.mat')
            Macaque2_M1_190527= {'emgs': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][0][0],
           'nChan': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][2][0][0],
           'sorted_isvalid': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][8],
           'sorted_resp': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][9],              
           'sorted_respMean': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][15],
           'ch2xy': Macaque2_M1_190527['Macaque2_M1_190527'][0][0][14],
           'DimSearchSpace': 96}
            SET=Macaque2_M1_190527
        return SET
    elif dataset_type=='rat':  # rat dataset has 6 subjects
        if m_i==0:
            rat1_M1_190716 = scipy.io.loadmat(path_to_dataset+'/rat1_M1_190716.mat')
            rat1_M1_190716= {'emgs': rat1_M1_190716['rat1_M1_190716'][0][0][0][0],
           'nChan': rat1_M1_190716['rat1_M1_190716'][0][0][2][0][0],
           'sorted_isvalid': rat1_M1_190716['rat1_M1_190716'][0][0][8],
           'sorted_resp': rat1_M1_190716['rat1_M1_190716'][0][0][9],              
           'sorted_respMean': rat1_M1_190716['rat1_M1_190716'][0][0][15],
           'ch2xy': rat1_M1_190716['rat1_M1_190716'][0][0][14],
           'DimSearchSpace': 32}            
            return rat1_M1_190716
        if m_i==1:
            rat2_M1_190617 = scipy.io.loadmat(path_to_dataset+'/rat2_M1_190617.mat')
            rat2_M1_190617= {'emgs': rat2_M1_190617['rat2_M1_190617'][0][0][0][0],
           'nChan': rat2_M1_190617['rat2_M1_190617'][0][0][2][0][0],
           'sorted_isvalid': rat2_M1_190617['rat2_M1_190617'][0][0][8],
           'sorted_resp': rat2_M1_190617['rat2_M1_190617'][0][0][9],              
           'sorted_respMean': rat2_M1_190617['rat2_M1_190617'][0][0][15],
           'ch2xy': rat2_M1_190617['rat2_M1_190617'][0][0][14],
           'DimSearchSpace': 32}         
            return rat2_M1_190617          
        if m_i==2:
            rat3_M1_190728 = scipy.io.loadmat(path_to_dataset+'/rat3_M1_190728.mat')
            rat3_M1_190728= {'emgs': rat3_M1_190728['rat3_M1_190728'][0][0][0][0],
           'nChan': rat3_M1_190728['rat3_M1_190728'][0][0][2][0][0],
           'sorted_isvalid': rat3_M1_190728['rat3_M1_190728'][0][0][8],
           'sorted_resp': rat3_M1_190728['rat3_M1_190728'][0][0][9],              
           'sorted_respMean': rat3_M1_190728['rat3_M1_190728'][0][0][15],
           'ch2xy': rat3_M1_190728['rat3_M1_190728'][0][0][14],
           'DimSearchSpace': 32}           
            return rat3_M1_190728                       
        if m_i==3:
            rat4_M1_191109 = scipy.io.loadmat(path_to_dataset+'/rat4_M1_191109.mat')
            rat4_M1_191109= {'emgs': rat4_M1_191109['rat4_M1_191109'][0][0][0][0],
           'nChan': rat4_M1_191109['rat4_M1_191109'][0][0][2][0][0],
           'sorted_isvalid': rat4_M1_191109['rat4_M1_191109'][0][0][8],
           'sorted_resp': rat4_M1_191109['rat4_M1_191109'][0][0][9],              
           'sorted_respMean': rat4_M1_191109['rat4_M1_191109'][0][0][15],
           'ch2xy': rat4_M1_191109['rat4_M1_191109'][0][0][14],
           'DimSearchSpace': 32}            
            return rat4_M1_191109                       
        if m_i==4:
            rat5_M1_191112 = scipy.io.loadmat(path_to_dataset+'/rat5_M1_191112.mat')
            rat5_M1_191112= {'emgs': rat5_M1_191112['rat5_M1_191112'][0][0][0][0],
           'nChan': rat5_M1_191112['rat5_M1_191112'][0][0][2][0][0],
           'sorted_isvalid': rat5_M1_191112['rat5_M1_191112'][0][0][8],
           'sorted_resp': rat5_M1_191112['rat5_M1_191112'][0][0][9],              
           'sorted_respMean': rat5_M1_191112['rat5_M1_191112'][0][0][15],
           'ch2xy': rat5_M1_191112['rat5_M1_191112'][0][0][14],
           'DimSearchSpace': 32}
                       
            return rat5_M1_191112                      
        if m_i==5:
            rat6_M1_200218 = scipy.io.loadmat(path_to_dataset+'/rat6_M1_200218.mat')        
            rat6_M1_200218= {'emgs': rat6_M1_200218['rat6_M1_200218'][0][0][0][0],
           'nChan': rat6_M1_200218['rat6_M1_200218'][0][0][2][0][0],
           'sorted_isvalid': rat6_M1_200218['rat6_M1_200218'][0][0][8],
           'sorted_resp': rat6_M1_200218['rat6_M1_200218'][0][0][9],              
           'sorted_respMean': rat6_M1_200218['rat6_M1_200218'][0][0][15],
           'ch2xy': rat6_M1_200218['rat6_M1_200218'][0][0][14],
           'DimSearchSpace': 32}          
            return rat6_M1_200218               
    elif dataset_type=='spinal':
        
        specific_subject = None
        match m_i:
            case 0:
                specific_subject = 'rat0_C5_500uA.pkl'
            case 1:
                specific_subject = 'rat1_C5_500uA.pkl'
            case 2:
                specific_subject = 'rat1_C5_700uA.pkl'
            case 3:
                specific_subject = 'rat1_midC4_500uA.pkl'
            case 4:
                specific_subject = 'rat2_C4_300uA.pkl'
            case 5:
                specific_subject = 'rat2_C5_300uA.pkl'
            case 6:
                specific_subject = 'rat2_C6_300uA.pkl'
            case 7:
                specific_subject = 'rat3_C4_300uA.pkl'
            case 8:
                specific_subject = 'rat3_C5_200uA.pkl'
            case 9:
                specific_subject = 'rat3_C5_350uA.pkl'
            case 10:
                specific_subject = 'rat3_C6_300uA.pkl'
        
         #load data
        with open(path_to_dataset + specific_subject, "rb") as f:
            data = pickle.load(f)
        
        ch2xy, emgs = data['ch2xy'], data['emgs']
        evoked_emg, filtered_emg = data['evoked_emg'], data['filtered_emg']
        maps = data['map']
        parameters = data['parameters']
        resp_region = data['resp_region']
        response = data['reponse']
        fs = data['sampFreqEMG']
        sorted_evoked = data['sorted_evoked']
        sorted_filtered = data['sorted_filtered']
        sorted_resp = data['sorted_resp']
        sorted_isvalid = data['sorted_isvalid']
        sorted_respMean = data['sorted_respMean']
        sorted_respSD = data['sorted_resp_SD']
        stim_channel = data['stim_channel']
        stimProfile=data['stimProfile']
        n_muscles = emgs.shape[0]

        #Computing baseline for filtered signal
        nChan = parameters['nChan'][0]
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 35 * 10**(-3))
        baseline = []
        for iChan in nChan:
            reps = np.where(stim_channel == iChan + 1)[0]
            n_rep = len(reps)
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, 0 : where_zero], axis=-1)
            baseline.append(mean_baseline)
        baseline = np.stack(baseline, axis=0)

        #remove baseline from filtered signal
        sorted_filtered[:, :, :n_rep, :] = sorted_filtered[:, :, :n_rep, :] - baseline[..., np.newaxis]
        sorted_resp = np.nanmax(sorted_filtered[:, :, :n_rep, int(resp_region[0]): int(resp_region[1])], axis=-1)
        masked_resp = np.ma.masked_where(sorted_isvalid[:, :, :n_rep] == 0, sorted_resp)
        sorted_respMean = masked_resp.mean(axis=-1)

         # compute baseline for evoked signal
        baseline = []
        for iChan in range(nChan):
            reps = np.where(stim_channel == iChan + 1)[0]
            n_rep = len(reps)
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_evoked[iChan, :, :n_rep, 0 : where_zero], axis=-1)
            baseline.append(mean_baseline)
        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)
        
        #remove baseline from evoked signal
        sorted_evoked[:, :, :n_rep, :] = sorted_evoked[:, :, :n_rep, :] - baseline[..., np.newaxis]
        sorted_resp = np.nanmax(sorted_evoked[:,:,:n_rep,int(resp_region[0]) :int(resp_region[1])], axis=-1)
        masked_resp = np.ma.masked_where(sorted_isvalid[:,:,:n_rep] == 0, sorted_resp)

        subject = {
            'emgs': emgs,
            'nChan': 64,
            'DimSearchSpace': 64,
            'sorted_respMean': sorted_respMean,
            'ch2xy': ch2xy,
            'dim_sizes': np.array([8, 4, 3, 4, 4]),
            'evoked_emg': evoked_emg, 'filtered_emg':filtered_emg, 'sorted_resp': sorted_resp,  
            'sorted_isvalid': sorted_isvalid, 'sorted_respSD': sorted_respSD,
            'sorted_filtered': sorted_filtered, 'stim_channel': stim_channel, 'fs': fs,
        'parameters': parameters, 'n_muscles': n_muscles, 'maps': maps,
        'resp_region': resp_region, 'stimProfile': stimProfile,  'baseline' : baseline    
        }
        
        return subject
        
    else:
        raise ValueError('The dataset type should be 5d_rat, nhp or rat' )

### --- BO method --- ###   

def neurostim_bo(dataset, model_cls, kappas):

    np.random.seed(0)

    # Experiment parameters initialization
    options = set_experiment(dataset)
    device = options['device']
    nRep = options['n_reps']
    nrnd = options['n_rnd']
    nSubjects = options['n_subjects']
    nEmgs = options['n_emgs']
    MaxQueries = options['n_Chan']
    ndims = options['n_dims']

    #Metrics initialization
    PP = torch.zeros((nSubjects,nEmgs,len(kappas),nRep, MaxQueries), device=DEVICE)
    PP_t = torch.zeros((nSubjects,nEmgs, len(kappas),nRep, MaxQueries), device=DEVICE)
    Q = torch.zeros((nSubjects,nEmgs,len(kappas),nRep, MaxQueries), device=DEVICE)
    Train_time = torch.zeros((nSubjects,nEmgs, len(kappas),nRep, MaxQueries), device=DEVICE)
    Cum_train =  torch.zeros((nSubjects,nEmgs, len(kappas),nRep, MaxQueries), device=DEVICE)
    PARTITIONS = np.empty((nSubjects,nEmgs,len(kappas),nRep, MaxQueries), dtype=object)
    RSQ = torch.zeros((nSubjects,nEmgs, len(kappas), nRep), device=DEVICE)
    REGRETS = torch.zeros((nSubjects,nEmgs, len(kappas), nRep, MaxQueries), device=DEVICE)
    SOBOLS = np.empty((nSubjects,nEmgs,len(kappas),nRep, MaxQueries), dtype=object)

    for s_idx in range(nSubjects):

        print(f'subject {s_idx + 1} \ {nSubjects}')
        subject = load_matlab_data(dataset, s_idx) 
        subject['ch2xy'] = subject['ch2xy'].to(device).clone().detach() if isinstance(subject['ch2xy'], torch.Tensor) else torch.tensor(subject['ch2xy'], device=device)

        for k_idx, kappa in enumerate(kappas):

            for e_i in range(len(subject['emgs'])):

                # "Ground truth" map
                if dataset == '5d_rat':
                    MPm= torch.mean(subject['sorted_respMean'].clone().detach().to(DEVICE)[:, s_idx % 4], axis=0)
                else:
                    MPm= torch.tensor(subject['sorted_respMean'][:,e_i]).float()  
                # Best known channel
                mMPm= torch.max(MPm)

                # priors and kernel handling
                priorbox = gpytorch.priors.SmoothedBoxPrior(a=math.log(options['rho_low']),b= math.log(options['rho_high']), sigma=0.01)
                outputscale_priorbox= gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01) 

                prior_lik= gpytorch.priors.SmoothedBoxPrior(a=options['noise_min']**2,b= options['noise_max']**2, sigma=0.01) # gaussian noise variance
                likf= gpytorch.likelihoods.GaussianLikelihood(noise_prior= prior_lik)
                likf.noise= [1.0]
                if device=='cuda':
                    likf=likf.cuda()

                # Metrics initialization
                # Then run the sequential optimization
                DimSearchSpace = subject['DimSearchSpace'] 
                #MaxQueries = DimSearchSpace
                perf_explore= torch.zeros((nRep, MaxQueries), device=device)
                perf_exploit= torch.zeros((nRep, MaxQueries), device=device)
                perf_rsq= torch.zeros((nRep), device=device)
                P_test =  torch.zeros((nRep, MaxQueries, 2), device=device) #storing all queries
                train_time = torch.zeros((nRep, MaxQueries), device=DEVICE)
                cum_time = torch.zeros((nRep, MaxQueries), device=DEVICE)
                partitions = np.empty((nRep, MaxQueries), dtype=object)
                regret = np.empty((nRep, MaxQueries), dtype=np.float32)
                sobol_interactions = np.empty((nRep, MaxQueries), dtype=object)


                for rep_i in range(nRep):
                    
                    # maximum response obtained in this round, used to normalize all responses between zero and one.
                    MaxSeenResp=0
                    q=0 # query number
                    timer = 0.0
                    order_this= torch.randperm(DimSearchSpace, device=device) # random permutation of each entry of the search space
                    P_max=[]

                    
                    executor = ThreadPoolExecutor(max_workers=2)
                    space_reconfiguration = None
                    interactions = None

                    while q < MaxQueries:
                        t0 = time.time()
                        # Query selection
                        if q>=nrnd:
                            # Max of acquisition map
                            AcquisitionMap = ymu + kappa*torch.nan_to_num(torch.sqrt(ys2)) # UCB acquisition
                            Next_Elec= torch.where(AcquisitionMap.reshape(len(AcquisitionMap))==torch.max(AcquisitionMap.reshape(len(AcquisitionMap))))
                            Next_Elec = Next_Elec[0][np.random.randint(len(Next_Elec))]  if len(Next_Elec[0]) > 1 else Next_Elec[0][0]
                            P_test[rep_i][q][0]= Next_Elec
                        else:
                            P_test[rep_i][q][0]= int(order_this[q])
                        query_elec = P_test[rep_i][q][0]

                        # Read response
                        if dataset == '5d_rat':
                            valid_resp = subject['sorted_respMean'][:, s_idx % 4, int(query_elec.item())]    
                        else:
                            valid_resp= torch.tensor(subject['sorted_resp'][int(query_elec)][e_i][subject['sorted_isvalid'][int(query_elec)][e_i]!=0])
                        r_i= np.random.randint(len(valid_resp))
                        test_respo= valid_resp[r_i]
                        test_respo += torch.normal(0.0, 0.02*torch.mean(test_respo))
                        # done reading response
                        P_test[rep_i][q][1]= test_respo

                        if (test_respo > MaxSeenResp) or (MaxSeenResp==0): # updated maximum response obtained in this round
                            MaxSeenResp=test_respo

                        x= subject['ch2xy'][P_test[rep_i][:q+1,0].long(),:].float() # search space position
                        x = x.reshape((len(x), ndims))
                        y= P_test[rep_i][:q+1,1]/MaxSeenResp # test result


                        # Model initialization and model update
                        if q == 0:
                            model = model_cls(x, y, likf, priorbox, outputscale_priorbox)
                            if model.name in ['neuralSobolGP', 'neuralMHGP']:
                                sobol = Sobol(dataset)
                                model.sobol = sobol #Initialize sobol
                                surrogate_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior= prior_lik)
                                surrogate = ExactGP(x, y, surrogate_likelihood, priorbox, outputscale_priorbox)
                                interactions = model.sobol.update_interactions(x, y, surrogate, surrogate_likelihood)
                        else:
                            model.set_train_data(x, y, strict=False)

                        # Model training
                        model.train()
                        likf.train()
                        model, likf, _ = optimize(model, x, y)


                        # Model evaluation
                        model.eval()
                        likf.eval()

                        with torch.no_grad():
                            X_test= subject['ch2xy'].float()
                            observed_pred = likf(model(X_test))

                            ymu= observed_pred.mean
                            ys2= observed_pred.variance

                        Tested= torch.unique(P_test[rep_i][:q+1,0]).long()
                        MapPredictionTested=ymu[Tested]
                        BestQuery = Tested if len(Tested) == 1 else Tested[(MapPredictionTested==torch.max(MapPredictionTested)).reshape(len(MapPredictionTested))]
                        if len(BestQuery) > 1:
                            BestQuery = np.array([BestQuery[np.random.randint(len(BestQuery))]])
                        
                        # Store metrics
                        P_max.append(BestQuery[0]) # Maximum response at time q
                        #msr[m_i,e_i,k_i,rep_i,q] = MaxSeenResp

                        # log regret calculations
                        chosen_pref = MPm[BestQuery].item()
                        instant_regret = mMPm.item() - chosen_pref
                        regret[rep_i, q] = instant_regret

                        # Pending Sobol job fetch
                        if space_reconfiguration is not None and space_reconfiguration.done():
                            try:
                                interactions = space_reconfiguration.result()
                                space_reconfiguration = None
                            except Exception as e:
                                space_reconfiguration = None
                                print(f'[Sobol] background job failed: {e}')
                            
                        #Update partitions every 10 queries
                        if model.name in ['neuralSobolGP', 'neuralMHGP']:
                            
                            partitions[rep_i, q] = model.partition
                            sobol_interactions[rep_i, q] = interactions.copy()

                            if q % 10 == 1:

                                if space_reconfiguration is None or space_reconfiguration.done():
                                    
                                    try:

                                        surrogate_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior= prior_lik)
                                        surrogate = ExactGP(x, y, surrogate_likelihood, priorbox, outputscale_priorbox)
                                        space_reconfiguration = executor.submit(model.reconfigure_space, surrogate, surrogate_likelihood)
                                
                                    except Exception as e:
                                        space_reconfiguration = None
                                        print(f"[Sobol] submit failed: {e}")
                            

                        # computation time calculations
                        t1 = time.time()
                        elapsed = t1 - t0
                        timer += elapsed
                        train_time[rep_i, q] = elapsed
                        cum_time[rep_i, q] = timer
                        q+=1
                    
                    executor.shutdown(wait=False)

                    # estimate current exploration performance: knowledge of best stimulation point
                    perf_explore[rep_i,:]=MPm[P_max].reshape((len(MPm[P_max])))/mMPm
                    # estimate current exploitation performance: knowledge of best stimulation point
                    perf_exploit[rep_i,:]= P_test[rep_i][:,0].long()
                    # R^2 correlation with ground truth
                    y_true = MPm
                    y_pred = ymu

                    ss_res = torch.sum((y_true - y_pred) ** 2)
                    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    perf_rsq[rep_i] = r_squared 
  

                # store all performance metrics
                PP[s_idx,e_i,k_idx]=perf_explore
                Q[s_idx,e_i,k_idx] = P_test[:,:,0]
                PP_t[s_idx,e_i,k_idx]= MPm[perf_exploit.long().cpu()]/mMPm
                Train_time[s_idx,e_i,k_idx] = train_time.mean(dim=0).detach().cpu()
                Cum_train[s_idx, e_i, k_idx] = cum_time.mean(dim=0).detach().cpu()
                PARTITIONS[s_idx,e_i,k_idx] = partitions
                RSQ[s_idx,e_i,k_idx]=perf_rsq
                REGRETS[s_idx,e_i,k_idx, :] = torch.log(torch.tensor(regret, dtype=torch.float32, device=DEVICE) + 1e-8) 
                SOBOLS[s_idx, e_i, k_idx] = sobol_interactions # mean_mats.mean(axis=0) #?# some mean operation

    # Saving variables
    output_dir = os.path.join('output', 'neurostim_experiments', dataset)
    os.makedirs(output_dir, exist_ok=True)
    fname = f'{dataset}_{model.name}_budget{MaxQueries}_{nRep}reps.npz'
    results_path = os.path.join(output_dir,fname)
    np.savez_compressed(results_path,
            RSQ=RSQ.cpu(), PP=PP.cpu(), PP_t=PP_t.cpu(), 
            kappas=np.array(kappas),
            PARTITIONS = PARTITIONS,
            SOBOLS = SOBOLS,
            REGRETS = REGRETS.cpu(),
            Train_time = Train_time.cpu(),
            Cum_train = Cum_train.cpu()
            )
    print(f'saved results to {results_path}')

    
### --- Parser handling

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

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='5d_rat', help='Neurostimulation dataset type')

    # Model selection
    parser.add_argument('--model_cls', type=str, default='ExactGPModel', help='Model class name (ExactGP, AdditiveGP, SobolGP, MHGP)')
   
    # Method-specific params
    parser.add_argument('--kappas', type=_parse_list_of_floats, default=None, help='Comma-separated kappas for kappa_search')

    # Misc
    parser.add_argument('--list_models', action='store_true')
    parser.add_argument('--list_methods', action='store_true')

    args = parser.parse_args(argv)

    # Allowed mappings (whitelist)
    model_map = {
        'ExactGPModel': ExactGPModel,
        'AdditiveKernelGP': AdditiveKernelGP,
        'SobolGP': neuralSobolGP,
        'MHGP': neuralMHGP,
    }

    if args.list_models:
        print('Available model classes:')
        for k in model_map.keys():
            print(' -', k)
        return

    if args.model_cls not in model_map:
        raise ValueError(f"Unknown model_cls '{args.model_cls}'. Use --list_models to see options.")

    model_cls = model_map[args.model_cls]

    # dispatch
    try:
        result = neurostim_bo(args.dataset, model_cls, kappas=args.kappas)
        
        print('Completed')

    except Exception as e:
        print('ERROR during execution:', e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    
    #neurostim_bo('rat', neuralSobolGP, kappas=[3.0, 5.0, 7.0, 9.0])
    #neurostim_bo('nhp', neuralSobolGP, kappas=[3.0, 5.0, 7.0, 9.0])
    #neurostim_bo('5d_rat', neuralSobolGP, kappas=[3.0, 5.0, 7.0, 9.0])
    main()