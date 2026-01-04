import gpytorch
import torch
import random
import copy
import numpy as np


from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gpytorch import GPyTorchModel


### Model utils ###

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


### Model classes ###

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
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

class MHGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, partition = None, history = None, sobol=None, epsilon=8e-2):

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean() 
        self.partition = partition if partition is not None else [[i] for i in range(train_x.shape[-1])]
        self.history = history if history else [self.partition_to_key(self.partition)] 
        self.sobol = sobol
        self.name = 'MHGP'
        self.n_dims = train_x.shape[-1]
        self.epsilon = 0.08 # - 0.02 * min(1.0, (self.n_dims**2 / 30.0))
        self.split_bias = 0.7
        self.max_attempts = self.bell_number(self.n_dims)

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
        return tuple(sorted(tuple(sorted(sub)) for sub in partition))

    def check_history(self, new_partition):
        """
        Return True if the canonical key for new_partition is already in history.
        """
        key = self.partition_to_key(new_partition)
        return key in self.history
    
    def update_partition(self, new_partition):

        # normalize indices to ints
        new_key = self.partition_to_key(new_partition)
        current_key = self.partition_to_key(self.partition)
        # avoid unnecessary rebuilds:
        if new_partition == current_key:
            return
        self.partition = [list(grp) for grp in new_key]
        self._build_covar()
    
    def sobol_partitioning(self, interactions):
        
        old_partition = self.partition
        has_candidate = False
        attempts = 0

        if len(self.history) == self.max_attempts:
            return old_partition

        while not has_candidate and attempts < (self.max_attempts + 5):

            new_partition = copy.deepcopy(self.partition)
            
            # Split strategy
            if random.random() < self.split_bias:

                robbed_idx = random.randint(0, len(new_partition) - 1)
                robbed_subset = new_partition[robbed_idx]

                if len(robbed_subset) > 1:

                    victim = np.random.choice(robbed_subset) #, random.randint(1, len(splitted_subset) - 1))
                    robbed_subset.remove(victim)
                    new_partition[robbed_idx] = robbed_subset # to fix split duplicate singletons issue
                    new_partition.append([victim.astype(int)]) 
                
                    if (not self.check_history(new_partition)) and self.are_additive(victim, robbed_subset, interactions):

                        has_candidate = True
                        return new_partition
                else:
                    attempts+=1

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
                    if has_candidate and (not self.check_history(new_partition)):
                        return new_partition
 
            attempts +=1

        #print(f'Number of attempts exceedded')
        return old_partition

    def bell_number(self, n):
        bell = [[0]*(n+1) for _ in range(n+1)]
        bell[0][0] = 1
        for i in range(1, n+1):
            bell[i][0] = bell[i-1][i-1]
            for j in range(1, i+1):
                bell[i][j] = bell[i-1][j-1] + bell[i][j-1]
        return bell[n][0]

    def are_additive(self, individual, set, interactions):

        for evaluator in set:

            i = min(evaluator, individual)
            j = max(evaluator, individual)

            if interactions[j] > self.epsilon:
                return False
        
        return True
        
    def calculate_acceptance(self, proposed_model, device='cpu'):

        self.train()
        self.likelihood.train()
        proposed_model.train()
        proposed_model.likelihood.train()

        mll_curr = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll_proposed = gpytorch.mlls.ExactMarginalLogLikelihood(proposed_model.likelihood, proposed_model)
        train_inputs = tuple(t.to(device) for t in self.train_inputs)

        curr_evidence = mll_curr(self(*train_inputs), self.train_targets)
        proposed_evidence = mll_proposed(proposed_model(proposed_model.train_inputs[0]), proposed_model.train_targets)

        acceptance = min(1.0, proposed_evidence / (curr_evidence + 1e-9))

        return acceptance
    
    def metropolis_hastings(self, interactions, device='cpu'):

        # update sobol interactions from surrogate model training
        new_partition = self.sobol_partitioning(interactions)
        if new_partition == self.partition:
            return self.partition
        proposed_model = MHGP(self.train_inputs[0], self.train_targets, gpytorch.likelihoods.GaussianLikelihood(), 
                              partition=new_partition, history=self.history, sobol=self.sobol)
        proposed_model.to(device)
        proposed_model.likelihood.to(device)
        proposed_model, proposed_model.likelihood, _ = optimize(proposed_model, proposed_model.train_inputs[0], proposed_model.train_targets)
        
        acceptance_ratio = self.calculate_acceptance(proposed_model)

        if acceptance_ratio >= 1.0:
            #print(f'Update for model accepted! Genetics propose to partition {new_partition} from {self.partition} with acceptance {acceptance_ratio}')
            self.history.append(self.partition_to_key(new_partition))
            return new_partition

        else: 
            return self.partition 
        
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
    def __init__(self, train_x, train_y, likelihood, partition=None, history = None, sobol=None, epsilon=8e-2):
        super(SobolGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.partition = partition if partition is not None else [[i] for i in range(train_x.shape[-1])]
        self.history = history if history else None
        self.sobol = sobol
        self.n_dims = train_x.shape[-1]
        self.epsilon = 0.08 #- 0.02 * min(1.0, (self.n_dims**2 / 30.0))
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
