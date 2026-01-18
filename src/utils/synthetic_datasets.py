import random
import numpy as np
from botorch.test_functions.synthetic import (
    Hartmann, Ackley, Griewank, Michalewicz, Rosenbrock, Shekel, Rastrigin, Levy, StyblinskiTang
)
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.sensitivity_analysis import Ishigami, Gsobol, Morris
import torch
import math
from typing import Optional, List, Literal


class SyntheticTestFun:

    def __init__(self, name, d, noise, negate,  *args, **kwargs):
        """Base constructor for synthetic test functions.

        Arguments:
            name: Name of BoTorch's test_functions among Hartmann, Ackley, Griewank, Michalewicz
            noise_std: Standard deviation of the observation noise.
        """
        self.d = d
        self.name = name
        self.device = torch.device("cpu")

        no_need_negate = ['twoblobs', 'dblobs', 'multprod', 'cyclical-fun', 'ishigami', 'gsobol', 'morris']
        negate = False if name in no_need_negate else True
        self.negate = negate


        match name:
            case 'hartmann':
                # Note: Need to negate to turn into maximization
                if d != 6:
                    raise ValueError("The Hartmann function needs to be 6 dimensional")
                self.f = Hartmann(noise_std=noise, negate=negate)
            case 'ackley':
                # Note: Need to negate to turn into maximization
                self.f = Ackley(d, noise_std=noise, negate=negate)
            case 'grienwank':
                # Note: Need to negate to turn into maximization
                self.f = Griewank(d, noise_std=noise, negate=negate)
            case 'michalewicz':
                # Note: Need to negate to turn into maximization
                self.f = Michalewicz(d, noise_std=noise, negate=negate)
            case 'ishigami':
                # Note: Keep original for maximization
                self.f = Ishigami(b=0.1, noise_std=noise, negate=negate)
            case 'gsobol':
                if d not in [6, 8, 15]:
                    raise ValueError("The GSobol function needs to be a 6, 8, or 15 dimensional")
                if d == 6:
                    a=[0, 0.5, 0.5, 99, 99, 99]
                elif d == 8:
                    a= [0, 1, 4.5, 9, 99, 99, 99, 99] 
                elif d == 15: 
                    a= [1, 2, 5, 10, 20, 50, 100, 500, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
                self.f = Gsobol(d, a, noise, negate=negate)
            case 'morris':
                if d != 20:
                    raise ValueError("The Morris function needs to be 20 dimensional")
                self.f = Morris(noise_std=noise, negate=negate)
            case 'schwefel':
                if d != 2:
                    raise ValueError("The Schewefel function needs to be a 2 dimensional")
                # Note: Keep original for maximization
                self.f = Schwefel(d, noise_std=noise, negate=negate)
            case 'powell':
                if d % 4 != 0:
                    raise ValueError("The Powell function needs to be a factor of 4 dimensional")
                # Note: Need to negate to turn into maximization
                self.f = Powell(d, noise_std=noise, negate=negate)
            case 'eggholder':
                if d != 2:
                    raise ValueError("The Eggholder function needs to be a 2 dimensional")
                # Note: Need to negate to turn into maximization
                self.f = EggholderFunction(noise_std=noise, negate=negate)
            case 'twoblobs':
                # Note: Keep original for maximization
                self.f = TwoBlobs(noise_std=noise, negate=False)
            case 'dblobs':
                # d-dimensional generalization: n_blobs defaults to d inside the class
                # Note: Keep original for maximization (like TwoBlobs)
                self.f = DBlobs(d=d, noise_std=noise, negate=False, *args, **kwargs)
                self.name = str(d) + '-' + self.name
            case 'rosenbrock':
                # Note: Need to negate to turn into maximization
                self.f = Rosenbrock(d, noise_std=noise, negate=negate)
            case 'rastrigin':
                # Note: Need to negate to turn into maximization
                self.f = Rastrigin(d, noise_std=noise, negate=negate)
            case 'levy':
                # Note: Need to negate to turn into maximization
                self.f = Levy(d, noise_std=noise, negate=negate)
            case 'shekel':
                # Note: Need to negate to turn into maximizationblin
                self.f = Shekel(m=5, negate=True)
                self.d = 4
            case 'styblinski':
                self.f = StyblinskiTang(d, noise_std=noise, negate=negate)
            case _:

                raise ValueError("Wrong synthetic function name.")


        self.f.d = d
        self.lower_bounds = np.array(self.f._bounds)[:, 0]
        self.upper_bounds = np.array(self.f._bounds)[:, 1]

    def to(self, device):
        """Moves the internal BoTorch function and object state to the specified device."""
        device = torch.device(device)
        self.device = device
        # BoTorch functions are nn.Modules, so they support .to()
        self.f.to(device) 
        return self

    def simulate(self, n_samples):
        """
        Simulate n_samples number of function calls to the test function.
        
        Returns: (X, Y) tuple of length n_samples containing those simulations.
        """

        dtype = torch.get_default_dtype()
        # convert bounds to torch tensors with correct dtype
        lb = torch.tensor(self.lower_bounds, dtype=dtype, device=self.device)
        ub = torch.tensor(self.upper_bounds, dtype=dtype, device=self.device)

        # draw uniform samples in [lb, ub] per-dimension
        X = lb + (ub - lb) * torch.rand(n_samples, self.d, dtype=dtype, device=self.device)

        # tensor management
        Y = self.f.forward(X)
        if isinstance(Y, torch.Tensor):
            Y = Y.to(dtype=dtype, device=self.device)
        else:
            Y = torch.tensor(np.asarray(Y), dtype=dtype, device=self.device)

        # shape management
        Y = Y.reshape(-1)

        return X, Y

class Schwefel(BaseTestProblem):
    """
    Schwefel function in standard minimization form:

      f_min(x) = 418.9828872724339 * d - sum_i x_i * sin(sqrt(|x_i|))

    - Domain: x_i in [-500, 500]
    - Global MINIMUM value: 0.0 at x_i ~= 420.9687
    """

    def __init__(self, d, noise_std=0.0, negate=False):
        self.d = d
        self._bounds = [(-500.0, 500.0)] * d
        self.noise_std = noise_std
        self.negate = negate
        self._offset = 418.9828872724339 * d # To turn into positive range, add + 1675.9305
        self.optimal_value = 0.0

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the *maximized* Schwefel function with optimal value = 0.
        """
        core = X * torch.sin(torch.sqrt(torch.abs(X)))
        y = self._offset - torch.sum(core, dim=-1)
        return y.reshape(-1)
    
    def forward(self, X):
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y += torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y

class Powell(BaseTestProblem) :
    def __init__(self, d, noise_std=0.0, negate=False):
        if d % 4 != 0:
            raise ValueError("Powell function requires dimension divisible by 4.")
        self.noise_std = noise_std
        self.negate = negate
        self.d = d
        self._bounds = [(-4.0, 5.0)] * d  # per-dimension bounds
        self.optimal_value = 0.0

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)
    
    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: tensor of shape (n_samples, d*4)
        Powell-term operates on each group of 4 dims independently,
        but the sum over blocks introduces 4‑way non-additive coupling per block.
        """
        def powell_block(x4):
            x1, x2, x3, x4v = x4[..., 0], x4[..., 1], x4[..., 2], x4[..., 3]
            term1 = (x1 + 10 * x2) ** 2
            term2 = 5 * (x3 - x4v) ** 2
            term3 = (x2 - 2 * x3) ** 4
            term4 = 10 * (x1 - x4v) ** 4
            return term1 + term2 + term3 + term4

        blocks = X.view(X.shape[0], self.d // 4 , 4)  
        fvals = torch.stack([powell_block(blocks[:, i, :]) for i in range(self.d // 4)], dim=1).sum(dim=1)

        return fvals.reshape(-1)

    def forward(self, X):
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y += torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y  

class EggholderFunction(BaseTestProblem):
    """
    Eggholder function (2D).
    f(x1,x2) = -(x2 + 47) * sin( sqrt(|x1/2 + (x2 + 47)|) )
                - x1 * sin( sqrt(|x1 - (x2 + 47)|) )
    Domain: x1, x2 in [-512, 512]
    Global minimum approx -959.6407 at (512, 404.2319)
    """
    def __init__(self, noise_std=0.0, negate=False):
        self._bounds = [(-512.0, 512.0), (-512.0, 512.0)]
        self.noise_std = noise_std
        self.negate = negate
        self._true_min = -959.6407 # with offset, it would be self._offset = 1048.4714
        self.optimal_value = -self._true_min if self.negate else self._true_min

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)
    
    def _evaluate_true(self, X):
        # X is (n,2)
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        term1 = -(x2 + 47.0) * torch.sin(torch.sqrt(torch.abs(x1 / 2.0 + (x2 + 47.0))))
        term2 = - x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47.0))))
        f_orig = term1 + term2
        return f_orig.reshape(-1)

    def forward(self, X):
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y

class TwoBlobs(BaseTestProblem):
    """
    Synthetic test function: weighted mixture of two 2D Gaussian blobs.
    One blob is scaled up so there is a clear single maximum at blob1's mean.
    """

    def __init__(self, noise_std=0.0, negate=False, weight1=1.0, weight2=0.4):
        
        # --- attributes BaseTestProblem expects BEFORE super().__init__() ---
        self.d = 2
        self.dim = self.d
        self.continuous_inds = list(range(self.d))
        self.discrete_inds = []
        self.categorical_inds = []
        self._bounds = [(0.0, 10.0) for _ in range(self.d)] # bounds on original domain

        # initialize BaseTestProblem / nn.Module internals
        super().__init__(noise_std, negate)

        self.noise_std = noise_std
        self.negate = negate

        dtype = torch.get_default_dtype()

        # Gaussian 1 (dominant blob)
        self.mean1 = torch.tensor([7.0, 6.0], dtype=dtype)
        self.std1 = torch.tensor([1.0, 0.5], dtype=dtype)
        self.rho1 = torch.tensor(0.7, dtype=dtype)
        self.weight1 = torch.tensor(weight1, dtype=dtype)

        # Gaussian 2 (smaller blob)
        self.mean2 = torch.tensor([4.0, 4.0], dtype=dtype)
        self.std2 = torch.tensor([0.7, 1.2], dtype=dtype)
        self.rho2 = torch.tensor(-0.3, dtype=dtype)
        self.weight2 = torch.tensor(weight2, dtype=dtype)

        # Peak value at mean1 (analytical)
        peak1 = self._gaussian_peak(self.std1, self.rho1) * self.weight1
        val_blob2_at_mean1 = self._gaussian_pdf_at_point(self.mean1, self.mean2, self.std2, self.rho2) * self.weight2
        self.optimal_value = (peak1 + val_blob2_at_mean1).item()


    def _gaussian_peak(self, std, rho):
        sx, sy = std[0], std[1]
        denom = 2 * math.pi * sx * sy * torch.sqrt(1 - rho**2)
        return 1.0 / denom

    def _gaussian_pdf_at_point(self, point, mean, std, rho):
        x, y = point[0], point[1]
        mx, my = mean[0], mean[1]
        sx, sy = std[0], std[1]

        denom = 2 * math.pi * sx * sy * torch.sqrt(1 - rho**2)
        Xnorm = (x - mx) / sx
        Ynorm = (y - my) / sy
        exp_term = ((Xnorm**2) - (2 * rho * Xnorm * Ynorm) + (Ynorm**2)) / (2 * (1 - rho**2))
        return torch.exp(-exp_term) / denom

    def gaussian_pdf(self, X, mean, std, rho):
        x = X[:, 0]
        y = X[:, 1]
        mx, my = mean[0], mean[1]
        sx, sy = std[0], std[1]

        denom = 2 * math.pi * sx * sy * torch.sqrt(1 - rho**2)
        Xnorm = (x - mx) / sx
        Ynorm = (y - my) / sy
        exp_term = ((Xnorm**2) - (2 * rho * Xnorm * Ynorm) + (Ynorm**2)) / (2 * (1 - rho**2))
        return torch.exp(-exp_term) / denom
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)

    def _evaluate_true(self, X):
        g1 = self.gaussian_pdf(X, self.mean1, self.std1, self.rho1) * self.weight1
        g2 = self.gaussian_pdf(X, self.mean2, self.std2, self.rho2) * self.weight2
        return g1 + g2

    def forward(self, X):
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y

class DBlobs(BaseTestProblem):
    """
    D-dimensional weighted mixture of multivariate Gaussian blobs (exchangeable corr).
    Several robustness features and two normalization modes included.

    Args:
        d: input dimension
        n_blobs: number of blobs (defaults to d)
        noise_std: observation noise std (added in forward)
        negate: whether to negate outputs (keeps shape semantics)
        seed: RNG seed
        rho_max: upper bound for exchangeable rho (default 0.4)
        normalize_method: 'max' (scale by estimated global max -> outputs in [0,1] approx)
                          or 'minmax' (map observed min->0 and max->1 from random sample)
        n_random_samples: budget for random sampling when searching (default 5000)
        refine_restarts, refine_steps, refine_lr: gradient-refine settings (used when normalize_search True & method='max')
        jitter: tiny jitter to ensure PD
    """

    def __init__(
        self,
        d: int = 5,
        n_blobs: Optional[int] = None,
        noise_std: float = 0.0,
        negate: bool = False,
        rho_max: float = 0.3,
        n_random_samples: int = 5000,
        refine_restarts: int = 8,
        refine_steps: int = 200,
        refine_lr: float = 0.05,
        jitter: float = 1e-8,
    ):
        # --- attributes BaseTestProblem expects BEFORE super().__init__() ---
        self.d = int(d)
        self.dim = self.d
        self.continuous_inds = list(range(self.d))
        self.discrete_inds = []
        self.categorical_inds = []
        self._bounds = [(0.0, 10.0) for _ in range(self.d)] # bounds on original domain


        # initialize BaseTestProblem / nn.Module internals
        super().__init__()

        # basic attributes
        self.noise_std = float(noise_std)
        self.negate = bool(negate)

        # set number of blobs
        self.n_blobs = n_blobs if n_blobs is not None else self.d

        # RNG
        torch.manual_seed(1)
        np.random.seed(1)

        # storage lists (will convert to tensors)
        means_list = []
        cov_list = []
        weights_list = []
        rhos_list = []

        dtype = torch.get_default_dtype()
        device = torch.device("cpu")

        # safe rho sampling range for exchangeable corr
        rho_upper = float(rho_max)
        rho_lower = -0.01

        # build blobs
        for i in range(self.n_blobs):
            mean = torch.as_tensor(2.0 + 6.0 * torch.rand(self.d), dtype=dtype, device=device)
            means_list.append(mean)

            stds = torch.as_tensor(0.5 + 1.5 * torch.rand(self.d), dtype=dtype, device=device)

            # sample rho in safe interval [rho_lower, rho_upper]
            rho_val = float(rho_lower + (rho_upper - rho_lower) * torch.rand(1).item())
            rhos_list.append(torch.as_tensor(rho_val, dtype=dtype, device=device))

            # covariance via exchangeable / compound-symmetry + PD regularization
            cov = self._create_exchangeable_covariance(stds, float(rho_val), eps=jitter)
            cov_list.append(cov)

            # initial weights (dominant first)
            weight = 1.0 if i == 0 else 0.7 / (i + 1)
            weights_list.append(float(weight))

        # force first blob to be dominant weight if desired
        weights_list[0] = float(max(weights_list[0], 1.0))

        # convert to stacked tensors
        self.means = torch.stack(means_list, dim=0)            # (n_blobs, d)
        self.cov_matrices = torch.stack(cov_list, dim=0)       # (n_blobs, d, d)
        self.weights = torch.as_tensor(weights_list, dtype=dtype, device=device)  # (n_blobs,)
        self.rhos = torch.stack(rhos_list, dim=0)              # (n_blobs,)

        # compute stable inverses and log-dets per blob using cholesky
        cov_invs = []
        cov_logdets = []
        for k in range(self.n_blobs):
            C = self.cov_matrices[k]
            Cj = C + torch.eye(self.d, dtype=C.dtype, device=C.device) * (1e-10) # add tiny jitter for numeric stability
            L = torch.linalg.cholesky(Cj) # Cholesky (should succeed as we regularized earlier)
            invC = torch.cholesky_inverse(L) 
            cov_invs.append(invC)

            # compute logdet robustly
            sign, slogdet = torch.linalg.slogdet(Cj)
            if sign <= 0:
                # fallback to eigen-clamp
                vals, vecs = torch.linalg.eigh(Cj)
                vals_clamped = torch.clamp(vals, min=1e-12)
                Cj2 = vecs @ torch.diag(vals_clamped) @ vecs.T
                sign, slogdet = torch.linalg.slogdet(Cj2)
            cov_logdets.append(float(slogdet))

        self.cov_invs = torch.stack(cov_invs, dim=0)          # (n_blobs, d, d)
        self.cov_logdets = torch.as_tensor(cov_logdets, dtype=dtype, device=device)  # (n_blobs,)

        # Analytical "value at dominant mean" (useful fallback)
        self.optimal_point = self.means[0].clone()
        analytic_val = float(self._evaluate_single_point(self.optimal_point.unsqueeze(0)).squeeze().item())
        self.optimal_value = analytic_val

        # ---------- Normalization logic ----------
        self._scale = max(float(analytic_val), 1e-12)  # fallback scale
        self._min_obs = 0.0
        self._max_obs = float(analytic_val)

        # random sampling budget; do in batches
        best_val = analytic_val
        worst_val = float("inf")
        device = self.means.device
        dtype = self.means.dtype
        n_rem = max(0, int(n_random_samples))
        batch = 4096
        # simple random sampling
        while n_rem > 0:
            bs = min(batch, n_rem)
            Xs = 10.0 * torch.rand(bs, self.d, dtype=dtype, device=device)
            Ys = self._evaluate_single_point(Xs).reshape(-1)
            max_Y = float(Ys.max().item())
            min_Y = float(Ys.min().item())
            if max_Y > best_val:
                best_val = max_Y
            if min_Y < worst_val:
                worst_val = min_Y
            n_rem -= bs

        # Refine best point via gradient ascent 
           
        best_ref_val = best_val
        best_ref_x = self.optimal_point.clone()
        starts = [self.optimal_point.clone()]
        for _ in range(refine_restarts - 1):
            starts.append(10.0 * torch.rand(self.d, dtype=dtype, device=device))

        for s in starts:
            x = s.clone().detach().to(dtype).requires_grad_(True)
            opt = torch.optim.Adam([x], lr=refine_lr)
            for _ in range(refine_steps):
                opt.zero_grad()
                v = self._evaluate_single_point(x.unsqueeze(0)).squeeze()
                (-v).backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(0.0, 10.0)
            vfin = float(self._evaluate_single_point(x.unsqueeze(0)).squeeze().item())
            if vfin > best_ref_val:
                best_ref_val = vfin
                best_ref_x = x.detach().clone()

        self._scale = float(max(best_ref_val, analytic_val, 1e-12))
        self.optimal_point = best_ref_x.clone()
        self._min_obs = float(min(worst_val, 0.0))
        self._max_obs = float(self._scale)
        self.optimal_value = 1.0

    # ----------------- helpers -----------------
    def _create_exchangeable_covariance(self, stds: torch.Tensor, rho: float, eps: float = 1e-9) -> torch.Tensor:
        """
        Compound-symmetry covariance:
           Sigma_ij = std_i * std_j * rho   (i != j)
           Sigma_ii = std_i^2
        Ensures PD by eigenvalue clamping if necessary.
        """
        d = stds.numel()
        outer = torch.outer(stds, stds)
        eye = torch.eye(d, dtype=stds.dtype, device=stds.device)
        cov = outer * (eye + (1.0 - eye) * float(rho))
        cov = 0.5 * (cov + cov.T)

        vals, vecs = torch.linalg.eigh(cov)
        vals_clamped = torch.clamp(vals, min=eps)
        cov_pd = vecs @ torch.diag(vals_clamped) @ vecs.T
        cov_pd = 0.5 * (cov_pd + cov_pd.T)
        cov_pd = cov_pd + torch.eye(d, dtype=cov_pd.dtype, device=cov_pd.device) * (1e-12)
        return cov_pd

    def _multivariate_gaussian_pdf(self, X: torch.Tensor, mean: torch.Tensor, cov_inv: torch.Tensor, cov_logdet: float) -> torch.Tensor:
        """
        Evaluate multivariate Gaussian pdf at rows of X for given mean, precision (cov_inv) and logdet.
        Returns (N,) for X shape (N,d) or scalar for (d,) inputs.
        """
        Xc = X - mean.unsqueeze(0)  # (N,d)
        quad = torch.einsum("ni,ij,nj->n", Xc, cov_inv, Xc)  # (N,)
        norm_log = -0.5 * (self.d * math.log(2.0 * math.pi) + float(cov_logdet))
        return torch.exp(norm_log - 0.5 * quad)  # (N,)

    def _evaluate_single_point(self, X: torch.Tensor) -> torch.Tensor:
        """
        Un-normalized mixture evaluation. Returns (N,) for batch X (N,d).
        """
        is_vec = X.dim() == 1
        if is_vec:
            X = X.unsqueeze(0)

        N = X.shape[0]
        result = torch.zeros(N, dtype=self.means.dtype, device=self.means.device)

        # sum weighted pdfs
        for k in range(self.n_blobs):
            pdf_vals = self._multivariate_gaussian_pdf(X, self.means[k], self.cov_invs[k], self.cov_logdets[k])
            result = result + float(self.weights[k]) * pdf_vals

        return result  # (N,)

    # main evaluation wrapper (normalized to [0,1] depending on scheme)
    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        vals = self._evaluate_single_point(X)  # (N,)
        # normalization
        scaled = vals / float(max(self._scale, 1e-12))
        scaled = torch.clamp(scaled, min=0.0, max=1.0)
        return scaled.reshape(-1)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper expected by some BaseTestProblem versions
        return self._evaluate_true(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y

    def get_blob_info(self) -> dict:
        """Return info about the blobs and normalization used."""
        return {
            "means": self.means,
            "weights": self.weights,
            "rhos": self.rhos,
            "cov_matrices": self.cov_matrices,
            "cov_invs": self.cov_invs,
            "cov_logdets": self.cov_logdets,
            "optimal_point": self.optimal_point,
            "optimal_value": self.optimal_value,
            "normalize_method": self._normalize_method,
            "scale": self._scale,
            "min_obs": self._min_obs,
            "max_obs": self._max_obs,
        }

    