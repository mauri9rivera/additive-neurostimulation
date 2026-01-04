import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import time
from concurrent.futures import ThreadPoolExecutor, wait
import cProfile
import pstats
import io
import os
import warnings
from linear_operator import settings

# Suppress numerical warnings for cleaner output
warnings.filterwarnings("ignore")

# -------------------------
# Data
# -------------------------
def make_data(n_train=1000, d=5, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n_train, d)
    true_w = torch.randn(d)
    y = (X * true_w).sum(dim=-1) + 0.1 * torch.randn(n_train)
    return X, y

# -------------------------
# Simple ExactGP Model (Standard GP)
# -------------------------
class SimpleExactGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SimpleExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# -------------------------
# Original Additive GP (Fixed)
# -------------------------
class OriginalAdditiveGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        input_dim = train_x.shape[-1]
        
        # Use RBF instead of Matern for better numerical stability
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([input_dim]),
                ard_num_dims=1,
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'OriginalAdditiveGP'

    def forward(self, X):
        mean = self.mean_module(X)
        batched_dimensions_of_X = X.mT.unsqueeze(-1)
        covar = self.covar_module(batched_dimensions_of_X).sum(dim=-3)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# -------------------------
# Corrected Batched Additive GP
# -------------------------
class BatchedAdditiveGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        input_dim = train_x.shape[-1]
        
        # Single batched kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([input_dim]),
                ard_num_dims=1,
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'BatchedAdditiveGP'
        self.input_dim = input_dim

    def forward(self, X):
        mean = self.mean_module(X)
        
        # Correct reshaping: [input_dim, n_data, 1]
        batched_X = X.transpose(0, 1).unsqueeze(-1)
        
        # Compute covariance in batch
        covar = self.covar_module(batched_X)
        
        # Sum over the batch dimension (dimensions)
        covar = covar.sum(dim=0)
        
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# -------------------------
# AdditiveStructureKernel GP (Corrected)
# -------------------------
class AdditiveStructureGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        input_dim = train_x.shape[-1]
        
        # Correct usage of AdditiveStructureKernel
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=torch.tensor([0]))
        additive_kernel = gpytorch.kernels.AdditiveStructureKernel(base_kernel, num_dims=input_dim)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(additive_kernel)
        self.likelihood = likelihood
        self.name = 'AdditiveStructureGP'

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# -------------------------
# Manual Additive Kernel (More Stable)
# -------------------------
class ManualAdditiveGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        input_dim = train_x.shape[-1]
        
        # Create separate kernels for each dimension
        self.kernels = torch.nn.ModuleList([
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=torch.tensor([i]))
            ) for i in range(input_dim)
        ])
        
        self.likelihood = likelihood
        self.name = 'ManualAdditiveGP'
        self.input_dim = input_dim

    def forward(self, x):
        mean_x = self.mean_module(x)
        
        # Sum covariances from all dimensions
        covar_x = self.kernels[0](x)
        for i in range(1, self.input_dim):
            covar_x = covar_x + self.kernels[i](x)
            
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# -------------------------
# Optimized Parallel Additive GP
# -------------------------
class OptimizedParallelAdditiveGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, max_workers=None):
        super().__init__(train_x, train_y, likelihood)
        input_dim = train_x.shape[-1]
        
        self.kernels = torch.nn.ModuleList([
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=torch.tensor([i]))
            ) for i in range(input_dim)
        ])
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'OptimizedParallelAdditiveGP'
        self.input_dim = input_dim
        self.max_workers = max_workers

    def _compute_single_covar(self, i, X):
        return self.kernels[i](X)

    def forward(self, X):
        mean = self.mean_module(X)
        
        covar_list = [None] * self.input_dim
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._compute_single_covar, i, X): i 
                for i in range(self.input_dim)
            }
            
            for future in future_to_index:
                idx = future_to_index[future]
                try:
                    covar_list[idx] = future.result()
                except Exception as exc:
                    covar_list[idx] = self._compute_single_covar(idx, X)
        
        total_covar = covar_list[0]
        for i in range(1, self.input_dim):
            total_covar = total_covar + covar_list[i]
            
        return gpytorch.distributions.MultivariateNormal(mean, total_covar)

# -------------------------
# ActiveDims Additive GP
# -------------------------
class ActiveDimsGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):

        super(ActiveDimsGP, self).__init__(train_x, train_y, likelihood)

        input_dim = train_x.shape[-1]
        self.likelihood = likelihood
        self.build_subgps(input_dim)
        
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def build_subgps(self, d):

        gps = []
        for i in range(d):
            subgp = SimpleExactGP(self.train_inputs[0][:, i].unsqueeze(-1), self.train_targets, self.likelihood)
            gps.append(subgp)
        
        self.gps = gps

    def submodel_forward(self, i, x):
        return self.gps[i].forward(x)

    def forward(self, x):

        input_dim = self.train_inputs[0].shape[-1]
        means = 0
        covars = 0
        with ThreadPoolExecutor(max_workers=input_dim) as executor:
            
            future_to_index = {
                executor.submit(self.submodel_forward, i, x): i
                for i in range(input_dim)
            }

            for future in future_to_index:
                idx = future_to_index[future]
                result = future.result()
                means += result.mean
                covars += result.covariance_matrix

        return gpytorch.distributions.MultivariateNormal(means, covars)




# -------------------------
# Model Factories
# -------------------------
def simple_exactgp_factory(X, y, likelihood):
    return SimpleExactGP(X, y, likelihood)

def original_additive_factory(X, y, likelihood):
    return OriginalAdditiveGP(X, y, likelihood)

def batched_additive_factory(X, y, likelihood):
    return BatchedAdditiveGP(X, y, likelihood)

def additive_structure_factory(X, y, likelihood):
    return AdditiveStructureGP(X, y, likelihood)

def manual_additive_factory(X, y, likelihood):
    return ManualAdditiveGP(X, y, likelihood)

def optimized_parallel_factory(X, y, likelihood, max_workers=None):
    return OptimizedParallelAdditiveGP(X, y, likelihood, max_workers=max_workers)

def active_dims_additive_factory(X, y, likelihood):
    return ActiveDimsGP(X, y, likelihood)

# -------------------------
# Benchmark Function with Numerical Stability
# -------------------------
def benchmark_model_factory(model_factory, X, y, n_warmup=1, n_time=50, lr=0.1, 
                           profile=False, model_name="", **factory_kwargs):
    
    # Increase CG iterations and tolerance for better numerical stability
    with settings.max_cg_iterations(2000), settings.cg_tolerance(1e-1):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = model_factory(X, y, likelihood, **factory_kwargs)
        model.train(); likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # Warm-up with gradient clipping for stability
        for _ in range(n_warmup):
            optimizer.zero_grad()
            out = model(X)
            loss = -mll(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Timing
        times = []
        for i in range(n_time):
            start = time.time()
            optimizer.zero_grad()
            out = model(X)
            loss = -mll(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            end = time.time()
            times.append(end - start)

        return times
    

def benchmark_parallel_model_factory(model_factory, X, y, n_warmup=1, n_time=50, lr=0.1, 
                           profile=False, model_name="", **factory_kwargs):
    
    # Increase CG iterations and tolerance for better numerical stability
    with settings.max_cg_iterations(2000), settings.cg_tolerance(1e-1):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = model_factory(X, y, likelihood, **factory_kwargs)
        model.train(); likelihood.train()
        optimizers = []
        mlls = []

        def train_submodel(submodel, optim, mll, X, y):
            submodel.train()
            submodel.likelihood.train()

            optim.zero_grad()
            out = submodel(X)
            loss = -mll(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(submodel.parameters(), max_norm=1.0)
            optimizer.step()


        for i, submodel in enumerate(model.gps):
            submodel.train()
            submodel.likelihood.train()
        
            optimizer = torch.optim.Adam(submodel.parameters(), lr=lr)

            mll = ExactMarginalLogLikelihood(submodel.likelihood, submodel)

            optimizers.append(optimizer)
            mlls.append(mll)

            x = model.train_inputs[0][:, i].unsqueeze(-1)

            train_submodel(submodel, optimizer, mll, x, y)

        # Timing
        times = []
        for i in range(n_time):
            start = time.time()

             # submit all futures
            with ThreadPoolExecutor(max_workers=len(model.gps)) as executor:
                futures = [
                    executor.submit(
                        train_submodel,
                        model.gps[j],
                        optimizers[j],
                        mlls[j],
                        model.train_inputs[0][:, j].unsqueeze(-1),
                        y
                    )
                    for j in range(len(model.gps))
                ]

                # wait for all to finish (raises any exceptions when you call result())
                wait(futures, return_when='ALL_COMPLETED')

            end = time.time()   # recorded only after wait returned and executor context closed
            times.append(end - start)

        return times

# -------------------------
# Run Analysis
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    X, y = make_data(n_train=20000, d=5, seed=42)
    
    print("=" * 60)
    print("CORRECTED GP MODEL ANALYSIS")
    print("=" * 60)
    print("Using RBF kernel for better numerical stability")
    print("Increased CG iterations to 2000")
    print("=" * 60)
    
    models_to_test = [
        ("Simple ExactGP (RBF)", simple_exactgp_factory, {}),
        ("Original AdditiveGP", original_additive_factory, {}),
        ("Batched AdditiveGP", batched_additive_factory, {}),
        ("AdditiveStructure GP", additive_structure_factory, {}),
        ("Manual AdditiveGP", manual_additive_factory, {}),
        ("Optimized Parallel (5 workers)", optimized_parallel_factory, {"max_workers": 5}),
        ("ActiveDims AdditiveGP", active_dims_additive_factory, {}),
    ]
    
    results = {}
    
    print("\nRunning corrected benchmarks...")
    for model_name, factory, kwargs in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        
        try:
            if model_name == "ActiveDims AdditiveGP":
                times = benchmark_parallel_model_factory(
                    factory, X, y, n_warmup=1, n_time=5, **kwargs
                )
            else:
                times = benchmark_model_factory(
                    factory, X, y, n_warmup=1, n_time=5, **kwargs
                )
            
            avg_time = sum(times) / len(times)
            results[model_name] = avg_time
            
            print(f"✓ Success: {avg_time:.4f}s per step")
            print(f"  Times: {[f'{t:.3f}' for t in times]}")
            
        except Exception as e:
            results[model_name] = None
            print(f"✗ Failed: {str(e)}")
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if successful_results:
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1])
        
        print("\nRanked by Performance:")
        for i, (model_name, avg_time) in enumerate(sorted_results):
            speedup = sorted_results[0][1] /  avg_time 
            print(f"{i+1:2d}. {model_name:30s}: {avg_time:.4f}s (relative: {speedup:.2f}x)")
    
    
    
