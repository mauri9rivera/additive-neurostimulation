import gpytorch
import math
import torch
import random
import copy
import numpy as np

from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
        self.likf = likelihood
        self.name = 'ExactGP'

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BaseGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims= 5), neural=False):
        super(BaseGP, self).__init__(train_x, train_y, likelihood)

        input_dim = train_x[0].shape[-1]

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)) if not neural else kernel
        self.likf = likelihood
        self.name = 'BaseGPBO'

    @property
    def num_outputs(self):
        return 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SimpleGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior=None, n_dims=5):
        super(SimpleGP, self).__init__(train_x, train_y, likelihood)

        scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)
        matk= gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims= n_dims, lengthscale_prior= lengthscale_prior) 
        matk_scaled = gpytorch.kernels.ScaleKernel(matk, outputscale_prior= scale_prior)
        matk_scaled.base_kernel.lengthscale = [1.0]*n_dims
        matk_scaled.outputscale=[1.0]

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = matk_scaled
        self.name = 'VanillaGPBO'

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class TemporoSpatialGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, lengthscale_prior = None, 
                ard_num_dims_group1=3, ard_num_dims_group2=2):
        super().__init__(train_x, train_y, likelihood)

        scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)
        
        self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel( 
                                nu=2.5, 
                                ard_num_dims=ard_num_dims_group1, 
                                active_dims=[0, 1, 2],
                                lengthscale_prior= lengthscale_prior),
                                outputscale_prior= scale_prior)
        self.temporal_kernel.base_kernel.lengthscale = [1.0]*ard_num_dims_group1
        self.temporal_kernel.outputscale=[1.0]

        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel(
                                nu=2.5, 
                                ard_num_dims=ard_num_dims_group2, 
                                active_dims=[3, 4],
                                lengthscale_prior= lengthscale_prior),
                                outputscale_prior=scale_prior)
        self.spatial_kernel.base_kernel.lengthscale = [1.0]*ard_num_dims_group2
        self.spatial_kernel.outputscale=[1.0]

        self.mean_module = gpytorch.means.ZeroMean()
        self.likelihood = likelihood
        self.name = 'TemporoSpatialGP'

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.temporal_kernel(x) + self.spatial_kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class TemporoSpatialGPv2(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, lengthscale_prior = None, 
                temp_dims=3, spatial_dims=2):
        super().__init__(train_x, train_y, likelihood)

        scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)
        
        self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel( 
                                nu=2.5, 
                                batch_shape=torch.Size([temp_dims]),
                                ard_num_dims=1, 
                                lengthscale_prior= lengthscale_prior),
                                outputscale_prior= scale_prior)
        self.temporal_kernel.base_kernel.lengthscale = [1.0]*temp_dims
        self.temporal_kernel.outputscale=torch.tensor([1.0]*temp_dims)
        self.temporal_dims= temp_dims
        self.spatial_dims=spatial_dims

        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel(
                                nu=2.5, 
                                ard_num_dims=spatial_dims, 
                                active_dims=[3, 4],
                                lengthscale_prior= lengthscale_prior),
                                outputscale_prior=scale_prior)
        self.spatial_kernel.base_kernel.lengthscale = [1.0]*spatial_dims
        self.spatial_kernel.outputscale=[1.0]

        self.mean_module = gpytorch.means.ZeroMean()
        self.likelihood = likelihood
        self.name = 'TemporoSpatialGPv2'

    def forward(self, x):
        mean_x = self.mean_module(x)

        x_temporal = x[:, :self.temporal_dims]
        batched_dims_of_x = x_temporal.mT.unsqueeze(-1)
        temporal_covar_batches = self.temporal_kernel(batched_dims_of_x)
        temporal_covar = temporal_covar_batches.sum(dim=-3)
        covar_x = temporal_covar + self.spatial_kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
  
class ParallelizedGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior=None, 
                ard_num_dims_group1=3, ard_num_dims_group2=2):
        super().__init__(train_x, train_y, likelihood)

        scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)
        
        self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel( 
                                mu=2.5, 
                                ard_num_dims=ard_num_dims_group1, 
                                active_dims=[0, 1, 2],
                                lengthscale_prior=lengthscale_prior),
                                outputscale_prior= scale_prior)
        self.temporal_kernel.base_kernel.lengthscale = [1.0]*ard_num_dims_group1
        self.temporal_kernel.outputscale=[1.0]

        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel(
                                mu=2.5, 
                                ard_num_dims=ard_num_dims_group2, 
                                active_dims=[3, 4],
                                lengthscale_prior=lengthscale_prior),
                                outputscale_prior=scale_prior)
        self.spatial_kernel.base_kernel.lengthscale = [1.0]*ard_num_dims_group2
        self.spatial_kernel.outputscale=[1.0]

        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'ParallelizedGP'

        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()

    def forward(self, x):
        mean_x = self.mean_module(x)

        '''# Parallelize kernel computations
        kernels = [self.temporal_kernel, self.spatial_kernel]
        covars = torch.nn.parallel.parallel_apply(kernels, [x, x])
        
        # Sum the covariance matrices
        covar_x = covars[0] + covars[1] 
        '''

        with torch.cuda.stream(self.stream1):
            covar_temp = self.temporal_kernel(x)

        with torch.cuda.stream(self.stream2):
            covar_spat = self.spatial_kernel(x)

        torch.cuda.synchronize()

        covar_x = covar_temp + covar_spat

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class AdditiveLearningGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior=None, 
                group_dims=[[0, 1, 2], [3, 4]]):
        super().__init__(train_x, train_y, likelihood)

        scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)

        self.group_dims = group_dims
        self.kernels = []
        for group in group_dims:

            kernel = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(
                            mu=2.5,
                            ard_num_dims= len(group),
                            active_dims=group,
                            lengthscale_prior=lengthscale_prior),
                        outputscale_prior= scale_prior)
            kernel.base_kernel.lengthscale = [1.0] * len(group)
            kernel.outputscale = [1.0]
            self.kernels.append(kernel)


        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'KernelLearningGP'

    def forward(self, x):
        mean_x = self.mean_module(x)

        print(f'device of input: {x.device}')
        for kernel in self.kernels:
            print(f'device of kernel: {kernel.device}')
        # Parallelize kernel computations
        covars = torch.nn.parallel.parallel_apply(self.kernels, [x for _ in self.kernels])
        
        # Sum the covariance matrices
        covar_x = sum(covars)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)   

class AdditiveKernelGP(gpytorch.models.ExactGP):

    def __init__(self, X_train, y_train, likelihood, lengthscale_prior=None, n_dims=5, neural=False):
        super().__init__(X_train, y_train, likelihood)

        if neural:
            scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)

            self.covar_module = gpytorch.kernels.ScaleKernel(
                                    gpytorch.kernels.MaternKernel(
                                        nu=2.5,
                                        batch_shape=torch.Size([n_dims]),
                                        ard_num_dims=1,
                                        lengthscale_prior=lengthscale_prior),
                                    outputscale_prior=scale_prior)
            self.covar_module.base_kernel.lengthscale = [1.0] * n_dims
            self.covar_module.outputscale = [1.0]

        else:
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
        self.name = 'GPYTorchAdditiveGP'

    @property
    def num_outputs(self):
        return 1

    def forward(self, X):
        mean = self.mean_module(X)
        batched_dimensions_of_X = X.mT.unsqueeze(-1)  # Now a d x n x 1 tensor
        covar = self.covar_module(batched_dimensions_of_X).sum(dim=-3)
        return gpytorch.distributions.MultivariateNormal(mean, covar) 
    
class GeneticAdditiveGP(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood, lengthscale_prior=None, d=5, 
                 num_models=3, budget=200):
        super().__init__(X_train, y_train, likelihood)

        self.acceptance_prob = lambda x: (0.6 + 0.4*(x / budget))
        self.lengthscale_prior = lengthscale_prior
        self.scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)

        self.n_dims = d
        self.num_models= num_models
        self.partitions = self.create_partitions()
        self.models = self.initialize_submodels()
        self.name = 'GeneticAdditiveGP'

    def create_partitions(self):

        partitions = [[random.sample(range(0, self.n_dims), int(self.n_dims // 2))] for i in range(self.num_models)]
        for p in partitions:
            p.append([x for x in range(self.n_dims) if x not in p[0]])

        print(f'These are the initial partitions: {partitions}')
        return partitions
    
    def initialize_submodels(self):

        models = []
        for partition in self.partitions:
            kernel = self.create_kernel(partition)
            model = BaseGP(self.train_inputs, self.train_targets, self.likelihood, kernel)
            model.covar_module = kernel
            models.append(model)

        return models
    
    def create_kernel(self, partition):

        kernels = []
        for dims in partition:
            kernel = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(
                            mu=2.5,
                            ard_num_dims=len(dims),
                            active_dims=dims,
                            lengthscale_prior = self.lengthscale_prior),
                            outputscale_prior=self.scale_prior)
            kernel.base_kernel.lengthscale = [1.0]*len(dims)
            kernel.outputscale=[1.0]
            kernels.append(kernel)

        combined_kernel = sum(kernels[1:], start=kernels[0]) ### TODO: Parallelization
        return combined_kernel

    def forward(self, x):

        means = 0.0
        covars = 0.0
        for model in self.models:
            pred = model.forward(x)
            means += pred.mean
            covars += pred.covariance_matrix

        means /= self.num_models
        covars += torch.eye(covars.size(0), device='cuda')*1e-3
        covars /= self.num_models

        return gpytorch.distributions.MultivariateNormal(means, covars)

    def simple_partitioning(self, model_idx):
        
        new_partition = copy.deepcopy(self.partitions[model_idx])
        old_partition = self.partitions[model_idx]

        sampling_prob = 0.01
        sampling_prob_inv = 0.00

        if random.random() < 0.5:
            #Split strategy

            splitted_subset = new_partition[random.randint(0, len(new_partition) - 1)]

            if len(splitted_subset) > 1:

                splitted_dim = np.random.choice(splitted_subset)
                splitted_subset.remove(splitted_dim)
                new_partition.append([splitted_dim.astype(int)])

                sampling_prob =  (1 / len(self.partitions)) * (1 / len(splitted_subset))
                sampling_prob_inv = (1 / math.comb(len(self.partitions)+1, 2))
        
        else:
            #Merge strategy

            if len(new_partition) > 1:

                robber_idx, robbed_idx = random.sample(range(len(new_partition)), 2)

                robbed_subset = new_partition[robbed_idx]
                robber_subset = new_partition[robber_idx]

                new_partition.remove(robbed_subset)
                new_partition.remove(robber_subset)
                new_partition.append(robbed_subset + robber_subset)

                sampling_prob =  (1 / math.comb(len(self.partitions), 2))
                sampling_prob_inv = (1 / len(self.partitions)-1) * (1 / len(robbed_subset))

        proposed_model = copy.deepcopy(self.models[model_idx])
        proposed_kernel = self.create_kernel(new_partition)
        proposed_model.covar_module = proposed_kernel

        proposal_ratio = sampling_prob_inv / sampling_prob
        
        return proposed_model, new_partition, proposal_ratio

    def calculate_acceptance(self, curr_model, proposed_model, proposal_ratio):

        mll_curr = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, curr_model)
        mll_proposed = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, proposed_model)

        curr_evidence = -mll_curr(curr_model(curr_model.train_inputs[0]), curr_model.train_targets)
        proposed_evidence = -mll_proposed(proposed_model(proposed_model.train_inputs[0]), proposed_model.train_targets)

        #print(f'Proposal ratio: {proposal_ratio} and evidence ratio: {proposed_evidence / curr_evidence}')

        acceptance = min(1, abs(proposal_ratio) * (proposed_evidence / curr_evidence))

        return acceptance

    def metropolis_hastings(self, iter, partition_strategy):

        models = []
        model_updates = torch.zeros((self.num_models), device='cuda')

        for i in range(self.num_models):

            current_model = self.models[i]
            proposed_model, new_partition, proposal_prob = partition_strategy(i)
            if proposal_prob == 0.0:
                models.append(current_model)
            else:
                proposed_model.cuda()
                proposed_model.likf.cuda()
                self.train_submodel(proposed_model)

                acceptance_ratio = self.calculate_acceptance(current_model, proposed_model, proposal_prob)
                if self.acceptance_prob(iter) < acceptance_ratio:
                    print(f'Update for {i}-th model accepted! Genetics propose to partition {new_partition} from {self.partitions[i]} with acceptance {acceptance_ratio}')
                    models.append(proposed_model)
                    self.partitions[i] = new_partition
                    model_updates[i] = 1
                else:
                    models.append(current_model)
            
        self.models = models
        return model_updates

    def propose_candidate(self, ch2xy, kappa):
        
        candidates = []
        for model in self.models:

            model.eval()
            model.likf.eval()

            with torch.no_grad():
                X_test = ch2xy
                observed_pred = model.likf(model.forward(X_test))

            MapPrediction = observed_pred.mean
            VariancePrediction = observed_pred.variance

            AcquisitionMap = MapPrediction + kappa*torch.nan_to_num(torch.sqrt(VariancePrediction)) # UCB acquisition
            NextQuery= torch.where(AcquisitionMap.reshape(len(AcquisitionMap))==torch.max(AcquisitionMap.reshape(len(AcquisitionMap))))
            NextQuery = NextQuery[0][np.random.randint(len(NextQuery[0]))] if len(NextQuery[0]) > 1 else NextQuery[0][0] # Multiple maximums case

            candidates.append(NextQuery)

        best_val = float('-inf')
        best_candidate = candidates[0]

        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            X_test = ch2xy
            observed_pred = self.likelihood(self.forward(X_test))
        MapPrediction = observed_pred.mean
        VariancePrediction = observed_pred.variance
        AcquisitionMap = MapPrediction + kappa*torch.nan_to_num(torch.sqrt(VariancePrediction)) # UCB acquisition
        
        for candidate in candidates:

            acq_val = AcquisitionMap[candidate.item()]
            if acq_val > best_val:
                best_candidate = candidate
                best_val = acq_val

        return best_candidate

    def train_submodels(self, n_iters=50):

        l = 0.0

        for model in self.models:
            model.train()
            model.likf.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likf, model)
            for _ in range(n_iters):

                output = model(model.train_inputs[0])
                loss = -mll(output, model.train_targets)
                l += loss
                loss.backward()
                optimizer.step()

        return l / self.num_models

    def train_submodel(self, submodel, n_iters=50):

        submodel.train()
        submodel.likf.train()
        optimizer = torch.optim.Adam(submodel.parameters(), lr=0.001)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(submodel.likf, submodel)
        for _ in range(n_iters):

            output = submodel(submodel.train_inputs[0])
            loss = -mll(output, submodel.train_targets)
            loss.backward()
            optimizer.step()

    def cuda_submodels(self):

        for model in self.models:
            model.cuda()
            model.likf.cuda()

    def set_train_data_submodels(self, x, y):

        self.set_train_data(x,y, strict=False)
        for model in self.models:

            model.set_train_data(x, y, strict=False)

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
