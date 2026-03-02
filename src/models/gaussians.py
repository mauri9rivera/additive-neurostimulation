import gpytorch
import torch
import random
import copy
import numpy as np
import math

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gpytorch import GPyTorchModel


### Model utils ###

def optimize(gp, train_x, train_y, n_iter=50, lr=0.01):
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
        gp (trained), likelihood (trained), epoch_losses (list of float)
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

    return gp, gp.likelihood, epoch_losses

def extract_lengthscales_log(model, d=None):
    """Extract per-dimension kernel lengthscales as a (d,) tensor in log-space.

    Returns a tensor where index i is the log-lengthscale for input dimension i,
    regardless of kernel structure (fully-coupled, batched-additive, or partitioned).
    """
    try:
        if hasattr(model.covar_module, 'kernels'):
            # AdditiveKernel (SobolGP, MHGP) — multiple ScaleKernel(MaternKernel)

            d = model.n_dims
            device = next(model.parameters()).device
            result = torch.zeros(d, device=device)
            for kernel in model.covar_module.kernels:
                ls = kernel.base_kernel.lengthscale.detach().flatten()
                active = kernel.base_kernel.active_dims
                for j, dim_idx in enumerate(active):
                    result[int(dim_idx)] = torch.log(ls[j])
            return result
        elif hasattr(model.covar_module, 'base_kernel'):
            # ExactGP or AdditiveGP — single ScaleKernel wrapping base kernel
            ls = model.covar_module.base_kernel.lengthscale.detach()
            return torch.log(ls.flatten())
        return None
    except Exception:
        return None

def eigf_scores(grid_points, gp_model, gp_likelihood):
    """
    Compute EIGF (Expected Improvement for Global Fit, Lam 2008) acquisition scores.

    Formula: phi_EIGF(x) = (f_hat(x) - y_nearest)^2 + sigma^2(x)

    Where y_nearest is the observed training output at the nearest training point to x
    (Voronoi neighbor). Pushes sampling toward regions where the surrogate disagrees
    with local observations OR is highly uncertain.

    Args:
        grid_points: Tensor (n_grid, d)
        gp_model: trained GPyTorch model
        gp_likelihood: corresponding likelihood

    Returns:
        scores: Tensor (n_grid,) of EIGF scores
    """
    train_x = gp_model.train_inputs[0]   # (n_train, d)
    train_y = gp_model.train_targets      # (n_train,)

    gp_model.eval()
    gp_likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = gp_likelihood(gp_model(grid_points))
        mean = output.mean       # (n_grid,)
        var = output.variance    # (n_grid,)

    dists = torch.cdist(grid_points, train_x)    # (n_grid, n_train)
    nearest_idx = dists.argmin(dim=1)             # (n_grid,)
    y_near = train_y[nearest_idx]                 # (n_grid,)

    return (mean - y_near) ** 2 + var             # (n_grid,)


def maximize_acq(kappa_val, gp_model, gp_likelihood, grid_points, mode='normal', acq='ucb'):
    """
    Grid-search acquisition function maximizer.
    Returns: new_x (1 x d tensor), acq_value (float), idx (int)

    Modes for kappa scheduling: 'normal', 'exponential_decay', 'log_growth'
    Acquisition functions: 'ucb' (default), 'eigf'
    """

    if acq == 'eigf':
        scores = eigf_scores(grid_points, gp_model, gp_likelihood)
        best_idx = torch.argmax(scores).item()
        best_x = grid_points[best_idx].unsqueeze(0)
        return best_x, scores[best_idx].item(), best_idx

    # UCB acquisition
    if mode == 'normal':
        pass
    elif mode == 'exponential_decay':
        t = gp_model.train_inputs[0].shape[0]
        kappa_val *= math.exp(-0.005*t)
    elif mode == 'log_growth':
        t = gp_model.train_inputs[0].shape[0]
        kappa_val = math.log((kappa_val**2)*t)

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


### Model classes ###

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.n_dims = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]))
        self.likf = likelihood
        self.name = 'ExactGP'

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class AdditiveGP(gpytorch.models.ExactGP):

    def __init__(self, X_train, y_train, likelihood):
        super().__init__(X_train, y_train, likelihood)
        input_dim = X_train.shape[-1]
        self.n_dims = input_dim
        self.covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel(
                                    nu=2.5,
                                    batch_shape=torch.Size([input_dim]),
                                    ard_num_dims=1,
                                )
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'AdditiveGP'

    @property
    def num_outputs(self):
        return 1

    def forward(self, X):
        mean = self.mean_module(X)
        batched_dimensions_of_X = X.mT.unsqueeze(-1)  # Now a d x n x 1 tensor
        covar = self.covar_module(batched_dimensions_of_X).sum(dim=-3)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class AdditiveDuvenaudGP(gpytorch.models.ExactGP):
    """
    Duvenaud-style additive GP (Duvenaud et al., 2011).

    Uses gpytorch.utils.sum_interaction_terms to compute the sum of all
    interaction terms up to `max_degree` from d univariate Matern-5/2 kernels.
    With max_degree=1 this is identical to AdditiveGP; with max_degree=2 it
    adds all pairwise interaction products k_i * k_j, etc.
    """

    def __init__(self, X_train, y_train, likelihood, max_degree=3):
        super().__init__(X_train, y_train, likelihood)
        input_dim = X_train.shape[-1]
        self.n_dims = input_dim
        self.max_degree = min(max_degree, input_dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                batch_shape=torch.Size([input_dim]),
                ard_num_dims=1,
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'AdditiveDuvenaudGP'

    @property
    def num_outputs(self):
        return 1

    def forward(self, X):
        mean = self.mean_module(X)
        batched_dimensions_of_X = X.mT.unsqueeze(-1)  # d x n x 1
        univariate_covars = self.covar_module(batched_dimensions_of_X)
        covar = gpytorch.utils.sum_interaction_terms(
            univariate_covars, max_degree=self.max_degree, dim=-3
        )
        # Newton-Girard subtraction can push near-zero eigenvalues negative;
        # add small diagonal jitter to keep the matrix numerically PSD.
        jitter = 1e-4 * torch.eye(covar.shape[-1], device=X.device, dtype=X.dtype)
        covar = covar + jitter
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class MHGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, partition=None, history=None, sobol=None):

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.partition = partition if partition is not None else [[i] for i in range(train_x.shape[-1])]
        self.history = history if history else [self.partition_to_key(self.partition)]
        self.sobol = sobol
        self.name = 'MHGP'
        self.n_dims = train_x.shape[-1]
        self.nb_partitions = self.n_dims

        # build covar_module based on partition
        self._build_covar()

    def max_singleton_partitions(self):
        """Number of distinct partitions reachable by _sample_partition.

        Each dimension independently chooses singleton vs. grouped, giving 2^d
        possibilities.  Use this to stop triggering Sobol reconfiguration once
        the history has explored all of them.
        """
        return 2 ** self.n_dims

    def history_exhausted(self):
        """Return True if the partition history covers all reachable partitions."""
        return len(self.history) >= self.max_singleton_partitions()

    def _build_covar(self):

        kernels = []
        for group in self.partition:

            ard_dims = len(group)

            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=ard_dims,
                active_dims=group,
            )

            scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel)

            kernels.append(scaled_kernel)

        self.covar_module = gpytorch.kernels.AdditiveKernel(*kernels)

    def partition_to_key(self, partition):
        return tuple(sorted(tuple(sorted(sub)) for sub in partition))

    def check_history(self, new_partition):
        """
        Return True if the canonical key for new_partition is already in history.
        """
        key = self.partition_to_key(new_partition)
        return key in self.history

    def update_partition(self, new_partition):

        # normalize indices to ints
        new_partition = [list(map(int, grp)) for grp in new_partition if len(grp) > 0]
        new_key = self.partition_to_key(new_partition)
        current_key = self.partition_to_key(self.partition)
        # avoid unnecessary rebuilds:
        if new_key == current_key:
            return
        self.partition = [list(grp) for grp in new_key]
        self._build_covar()

    def rank_and_select_partition(self, sobol_dict, lengthscales, device='cpu'):
        """
        Rank-based partition selection using Sobol total-order indices and log-lengthscales.

        1. Compute scores[i] = log(l_i) - ST_i  (higher = more likely additive/singleton)
        2. Softmax scores -> probs[i] = probability dimension i is a singleton
        3. Generate nb_partitions candidate partitions via _sample_partition(probs)
        4. Include current partition as extra candidate
        5. Filter out candidates already in history
        6. Evaluate candidates via parallel MLL -> return best partition

        Returns
        -------
        best_partition : list of lists
        """
        ST = np.asarray(sobol_dict['ST']).flatten()

        if lengthscales is not None:
            ls_np = lengthscales.detach().cpu().numpy().flatten()
        else:
            ls_np = np.zeros(self.n_dims)

        # Score: high log-lengthscale (insensitive) and low ST -> additive
        scores = ls_np - ST

        # Softmax to get per-dimension singleton probability
        scores_shifted = scores - scores.max()
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / (exp_scores.sum() + 1e-9)

        # Generate candidate partitions
        candidates = []
        candidates.append(self.partition) # Always include current partition as candidate

        for _ in range(self.nb_partitions):
            cand = self._sample_partition(probs)
            if not self.check_history(cand) and cand not in candidates:
                candidates.append(cand)



        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = self.partition_to_key(c)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        candidates = unique_candidates

        if len(candidates) == 1:
            return self.partition

        # Evaluate candidates in parallel
        best_partition, best_mll = self._evaluate_candidates_parallel(candidates, device)
        self.history.append(self.partition_to_key(best_partition))

        return best_partition

    def _sample_partition(self, probs):
        """
        Sample a partition from per-dimension singleton probabilities.

        For each dimension i: with probability probs[i], make it a singleton [i];
        otherwise add to a 'grouped' bucket.
        Returns partition as list-of-lists (singletons + one grouped list if non-empty).
        """
        singletons = []
        grouped = []
        for i in range(self.n_dims):
            if random.random() < probs[i]:
                singletons.append([i])
            else:
                grouped.append(i)

        partition = singletons[:]
        if grouped:
            partition.append(grouped)

        # Edge case: if all dimensions are grouped, return as single group
        if not partition:
            partition = [list(range(self.n_dims))]

        return partition

    def _evaluate_candidates_parallel(self, candidates, device='cpu'):
        """
        Evaluate candidate partitions by training a temporary MHGP and computing MLL.
        Uses ThreadPoolExecutor for parallel evaluation.

        Returns (best_partition, best_mll).
        """
        from concurrent.futures import ThreadPoolExecutor

        train_x = self.train_inputs[0]
        train_y = self.train_targets

        def eval_candidate(partition):
            try:
                lik = gpytorch.likelihoods.GaussianLikelihood().to(device)
                candidate_model = MHGP(train_x, train_y, lik, partition=partition).to(device)
                candidate_model, lik, _ = optimize(candidate_model, train_x, train_y, n_iter=30)

                candidate_model.train()
                lik.train()
                mll_fn = gpytorch.mlls.ExactMarginalLogLikelihood(lik, candidate_model)
                output = candidate_model(train_x)
                mll_val = mll_fn(output, train_y).item()
                return partition, mll_val
            except Exception:
                return partition, float('-inf')

        max_workers = min(len(candidates), 4)
        best_partition = self.partition
        best_mll = float('-inf')

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(eval_candidate, c) for c in candidates]
            for f in futures:
                partition, mll_val = f.result()
                if mll_val > best_mll:
                    best_mll = mll_val
                    best_partition = partition

        return best_partition, best_mll

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
    def __init__(self, train_x, train_y, likelihood, partition=None, history = None, sobol=None):
        super(SobolGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.partition = partition if partition is not None else [[i] for i in range(train_x.shape[-1])]
        self.history = history if history else None
        self.sobol = sobol
        self.n_dims = train_x.shape[-1]
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
        interactions = self.sobol.method(surrogate.train_inputs[0], surrogate.train_targets, surrogate)
        new_partition = self.sobol.update_partition(interactions)
        #self.update_partition(new_partition)

        return interactions, new_partition

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class WrappedModel(GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, model_cls, bounds=None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__()
        self.likelihood = likelihood
        self.train_y = train_y.squeeze(-1)
        self.train_x = train_x
        self.model = model_cls(train_x, self.train_y, self.likelihood, domain_bounds=bounds) if bounds is not None else model_cls(train_x, self.train_y, self.likelihood)
        self.model.train()
        self.likelihood.train()
        

    def fit(self, training_iter=50):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(training_iter):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y).mean()

            loss.backward()
            optimizer.step()

    def posterior(self, X, posterior_transform=None):
        self.model.eval()
        self.likelihood.eval()
        preds = self.likelihood(self.model(X))
        return preds
    
    def update_data(self, new_train_x, new_train_y):
        self.train_x = new_train_x
        self.train_y = new_train_y
        self.model.set_train_data(new_train_x, new_train_y, strict=False)

### Model for neurostimulation tasks ###
class NeuralExactGP(gpytorch.models.ExactGP):                                                                                                                                                                  
    """                                                                                                                                                                                                        
    ExactGP variant for neurostimulation datasets with configurable priors.                                                                                                                                    
    """                                                                                                                                                                                                        
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior=None, outputscale_prior=None):                                                                                                          
        super(NeuralExactGP, self).__init__(train_x, train_y, likelihood)                                                                                                                                      
        self.mean_module = gpytorch.means.ZeroMean()                                                                                                                                                           
        self.n_dims = train_x.shape[-1]                                                                                                                                                                        
        kernel = gpytorch.kernels.ScaleKernel(                                                                                                                                                                 
            gpytorch.kernels.MaternKernel(                                                                                                                                                                     
                nu=2.5, ard_num_dims=self.n_dims,                                                                                                                                                              
                lengthscale_prior=lengthscale_prior                                                                                                                                                            
            ),                                                                                                                                                                                                 
            outputscale_prior=outputscale_prior                                                                                                                                                                
        )                                                                                                                                                                                                      
        kernel.base_kernel.lengthscale = [1.0] * self.n_dims                                                                                                                                                   
        kernel.outputscale = [1.0]                                                                                                                                                                             
        self.covar_module = kernel                                                                                                                                                                             
        self.name = 'NeuralExactGP'                                                                                                                                                                            
                                                                                                                                                                                                               
    def forward(self, x):                                                                                                                                                                                      
        mean_x = self.mean_module(x)                                                                                                                                                                           
        covar_x = self.covar_module(x)                                                                                                                                                                         
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)                                                                                                                                      
                                                                                                                                                                                                                                                                                                                                                                                                                            
class NeuralAdditiveGP(gpytorch.models.ExactGP):                                                                                                                                                               
    """                                                                                                                                                                                                        
    AdditiveGP variant for neurostimulation datasets with configurable priors.                                                                                                                                 
    """                                                                                                                                                                                                        
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior=None, outputscale_prior=None):                                                                                                          
        super().__init__(train_x, train_y, likelihood)                                                                                                                                                         
        self.n_dims = train_x.shape[-1]                                                                                                                                                                        
                                                                                                                                                                                                               
        kernel = gpytorch.kernels.ScaleKernel(                                                                                                                                                                 
            gpytorch.kernels.MaternKernel(                                                                                                                                                                     
                nu=2.5, ard_num_dims=1, batch_shape=torch.Size([self.n_dims]),                                                                                                                       
                lengthscale_prior=lengthscale_prior                                                                                                                                                            
            ),                                                                                                                                                                                                 
            outputscale_prior=outputscale_prior                                                                                                                                                                
        )                                                                                                                                                                                                      
        kernel.base_kernel.lengthscale = [1.0] * self.n_dims                                                                                                                                                   
        kernel.outputscale = [1.0]                                                                                                                                                                             
        self.covar_module = kernel                                                                                                                                                                             
                                                                                                                                                                                                               
        self.mean_module = gpytorch.means.ZeroMean()                                                                                                                                                           
        self.name = 'NeuralAdditiveGP'                                                                                                                                                                         
                                                                                                                                                                                                               
    def forward(self, X):                                                                                                                                                                                      
        mean = self.mean_module(X)                                                                                                                                                                             
        batched_dimensions_of_X = X.mT.unsqueeze(-1)  # Now a d x n x 1 tensor                                                                                                                                 
        covar = self.covar_module(batched_dimensions_of_X).sum(dim=-3)                                                                                                                                         
        return gpytorch.distributions.MultivariateNormal(mean, covar)                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                           
class NeuralSobolGP(gpytorch.models.ExactGP):                                                                                                                                                                  
    """                                                                                                                                                                                                        
    SobolGP variant for neurostimulation datasets with configurable priors.                                                                                                                                    
                                                                                                                                                                                                               
    - partition: list of lists of integer dimension indices, e.g. [[0,2],[1,3]]                                                                                                                                
    - sobol: an associated NeuralSobol object (optional)                                                                                                                                                       
    - lengthscale_prior, outputscale_prior: GPyTorch priors for kernel hyperparameters                                                                                                                         
    """                                                                                                                                                                                                        
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior=None, outputscale_prior=None,                                                                                                           
                 partition=None, history=None, sobol=None):                                                                                                                                                    
        super(NeuralSobolGP, self).__init__(train_x, train_y, likelihood)                                                                                                                                      
        self.mean_module = gpytorch.means.ZeroMean()                                                                                                                                                           
        self.partition = partition if partition is not None else [[i] for i in range(train_x.shape[-1])]                                                                                                       
        self.history = history if history else None                                                                                                                                                            
        self.sobol = sobol                                                                                                                                                                                     
        self.n_dims = train_x.shape[-1]                                                                                                                                                                        
        self.name = 'NeuralSobolGP'                                                                                                                                                                            
        self.lengthscale_prior = lengthscale_prior                                                                                                                                                             
        self.outputscale_prior = outputscale_prior                                                                                                                                                             
        # build covar_module based on partition                                                                                                                                                                
        self._build_covar()                                                                                                                                                                                    
                                                                                                                                                                                                               
    def _build_covar(self):                                                                                                                                                                                    
        kernels = []                                                                                                                                                                                           
        for group in self.partition:                                                                                                                                                                           
            ard_dims = len(group)                                                                                                                                                                              
            base_kernel = gpytorch.kernels.MaternKernel(                                                                                                                                                       
                nu=2.5,                                                                                                                                                                                        
                ard_num_dims=ard_dims,                                                                                                                                                                         
                active_dims=group,                                                                                                                                                                             
                lengthscale_prior=self.lengthscale_prior                                                                                                                                                       
            )                                                                                                                                                                                                  
            scaled_kernel = gpytorch.kernels.ScaleKernel(                                                                                                                                                      
                base_kernel,                                                                                                                                                                                   
                outputscale_prior=self.outputscale_prior                                                                                                                                                       
            )                                                                                                                                                                                                  
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
        interactions = self.sobol.method(surrogate.train_inputs[0], surrogate.train_targets, surrogate)                                                                                                        
        new_partition = self.sobol.update_partition(interactions)                                                                                                                                              
        return interactions, new_partition                                                                                                                                                                     
                                                                                                                                                                                                               
    def forward(self, x):                                                                                                                                                                                      
        mean_x = self.mean_module(x)                                                                                                                                                                           
        covar_x = self.covar_module(x)                                                                                                                                                                         
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 
