import random
import numpy as np
from botorch.test_functions.synthetic import Hartmann, Ackley, Griewank, Michalewicz, Rosenbrock, Shekel
from botorch.test_functions.base import BaseTestProblem
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
        match name:
            case 'hartmann':
                # Note: Need to negate to turn into maximization
                if d != 6:
                    raise ValueError("The HartMann function needs to be 6 dimensional")
                self.f = Hartmann(noise_std=noise, negate=negate)
            case 'ackley':
                # Note: Keep original for maximization
                self.f = Ackley(d, noise_std=noise, negate=negate)
            case 'grienwank':
                # Note: Keep original for maximization
                self.f = Griewank(d, noise_std=noise, negate=negate)
            case 'michalewicz':
                # Note: Need to negate to turn into maximization
                self.f = Michalewicz(d, noise_std=noise, negate=negate)
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
                self.f = Rosenbrock(d, noise_std=noise, negate=negate)
            case 'shekel':
                self.f = Shekel(m=5, negate=True)
                self.d = 4
            case 'goldstein-price':
                self.f = GoldsteinPrice(noise_std=noise, negate=negate)
            case 'multprod':
                # kwargs: split=int|'half', u_type, v_type, alpha, rescale_positive, seed
                self.f = MultiplicativeInteraction(d=d, noise_std=noise, negate=False, **kwargs)
            case 'rosenbrock_rotated':
                # kwargs: rotation_seed
                self.f = RosenbrockRotated(dim=d, noise_std=noise, negate=negate, **kwargs)
                
            case 'ackley_correlated':
                # kwargs: correlation_strength (default 0.15)
                self.f = AckleyCorrelated(dim=d, noise_std=noise, negate=negate, **kwargs)
                
            case 'griewank_rosenbrock_hybrid':
                self.f = GriewankRosenbrockHybrid(dim=d, noise_std=noise, negate=negate, **kwargs)

            case 'cyclical-fun':
                # kwargs: poly_degree, poly_coeffs, trig, trig_freq, trig_amp, trig_phase, poly_scale, seed
                self.f = CyclicalFunction(d=d, noise_std=noise, negate=negate, **kwargs)

            case _:

                raise ValueError("Wrong synthetic function name.")


        self.f.d = d
        self.negate = False if name in ('twoblobs', 'dblobs', 'multprod', 'cyclical-fun') else negate
        self.lower_bounds = np.array(self.f._bounds)[:, 0]
        self.upper_bounds = np.array(self.f._bounds)[:, 1]


    def simulate(self, n_samples):
        """
        Simulate n_samples number of function calls to the test function.
        
        Returns: (X, Y) tuple of length n_samples containing those simulations.
        """

        dtype = torch.get_default_dtype()
        # convert bounds to torch tensors with correct dtype
        lb = torch.tensor(self.lower_bounds, dtype=dtype)
        ub = torch.tensor(self.upper_bounds, dtype=dtype)

        # draw uniform samples in [lb, ub] per-dimension
        X = lb + (ub - lb) * torch.rand(n_samples, self.d, dtype=dtype)

        # tensor management
        Y = self.f.forward(X)
        if isinstance(Y, torch.Tensor):
            Y = Y.to(dtype=dtype)
        else:
            Y = torch.tensor(np.asarray(Y), dtype=dtype)

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

    def __init__(self, noise_std=0.0, negate=False, weight1=1.0, weight2=0.3):
        self._bounds = [(0.0, 10.0), (0.0, 10.0)]
        self.noise_std = noise_std
        self.negate = negate
        self.d = 2

        dtype = torch.get_default_dtype()

        # Gaussian 1 (dominant blob)
        self.mean1 = torch.tensor([7.0, 6.0], dtype=dtype)
        self.std1 = torch.tensor([1.0, 0.5], dtype=dtype)
        self.rho1 = torch.tensor(0.3, dtype=dtype)
        self.weight1 = torch.tensor(weight1, dtype=dtype)

        # Gaussian 2 (smaller blob)
        self.mean2 = torch.tensor([4.0, 4.0], dtype=dtype)
        self.std2 = torch.tensor([0.7, 1.2], dtype=dtype)
        self.rho2 = torch.tensor(-0.4, dtype=dtype)
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

class DBlobs_old(BaseTestProblem):
    """
    D-dimensional generalization of TwoBlobs: weighted mixture of n Gaussian blobs.
    
    The function is a weighted sum of multivariate Gaussian PDFs in d-dimensions.
    One blob is scaled to be the dominant maximum.
    
    - Domain: x_i ∈ [0, 10] for all dimensions (consistent with TwoBlobs)
    - Global MAXIMUM value: analytically computed at the dominant blob's mean
    """

    def __init__(self, d: int = 5, n_blobs: Optional[int] = None, noise_std: float = 0.0, 
                 negate: bool = False, seed: Optional[int] = None):
        self.dim = d
        self._bounds = [(0.0, 10.0) for _ in range(d)]
        self.noise_std = noise_std
        self.negate = negate
        
        # Set number of blobs (default to d if not specified)
        self.n_blobs = n_blobs if n_blobs is not None else d
        
        # Random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize blobs
        self.means = []
        self.cov_matrices = []
        self.weights = []
        self.rhos = []  # Correlation parameters

        jitter = 1e-8
        
        # Generate random blobs
        for i in range(self.n_blobs):
            # Random mean within [2, 8] to avoid edges
            mean = torch.as_tensor(2.0 + 6.0 * torch.rand(d), dtype=torch.get_default_dtype())
            self.means.append(mean)
            
            # Random standard deviations between [0.5, 2.0]
            stds = torch.as_tensor(0.5 + 1.5 * torch.rand(d), dtype=torch.get_default_dtype())

            # Random correlation parameter rho for correlation structure
            rho_max = 0.4
            rho_min = -1.0 / float(d - 1) + 1e-6  # safe lower bound for exchangeable corr
            if rho_min >= rho_max:
                rho_min = rho_max - 1e-6 
            rho_val = float(rho_min + (rho_max - rho_min) * torch.rand(1).item())
            rho = torch.as_tensor(rho_val, dtype=torch.get_default_dtype())
            self.rhos.append(rho)
            
            # Create covariance matrix with exchangeable correlation structure
            cov_matrix = self._create_exchangeable_covariance(stds, rho, eps=jitter)
            self.cov_matrices.append(cov_matrix)
            
            # Weights: first blob gets weight 1.0, others get decreasing weights
            weight = 1.0 if i == 0 else 0.7 / (i + 1)
            self.weights.append(torch.tensor(weight))
        
        # Ensure first blob is dominant by giving it the highest weight
        self.weights[0] = torch.tensor(1.0)
        
        # Convert to tensors
        self.means = torch.stack(self.means)
        self.cov_matrices = torch.stack(self.cov_matrices)
        self.weights = torch.tensor(self.weights)
        self.rhos = torch.tensor(self.rhos)
        
        # Precompute covariance inverses and determinants for efficiency
        self.cov_dets = torch.linalg.det(self.cov_matrices)
        self.cov_invs = torch.linalg.inv(self.cov_matrices)
        
        # Analytical optimum is at the mean of the dominant (first) blob
        self.optimal_point = self.means[0]
        self.optimal_value = self._evaluate_single_point(self.optimal_point.unsqueeze(0)).item()

    def _create_exchangeable_covariance(self, stds: torch.Tensor, rho: float, eps: float = 1e-8) -> torch.Tensor:
        """
        Create covariance matrix with exchangeable correlation structure.
        
        Sigma_ij = std_i * std_j * rho for i ≠ j
        Sigma_ii = std_i²
        """
        d = stds.numel()
        # basic construction
        outer = torch.outer(stds, stds)  # std_i * std_j
        eye = torch.eye(d, dtype=stds.dtype, device=stds.device)
        cov = outer * (eye + (1.0 - eye) * rho)

        # symmetrize for safety
        cov = 0.5 * (cov + cov.T)

        # attempt a Cholesky; if it fails, do eigen-decomposition and clamp eigenvalues
        try:
            torch.linalg.cholesky(cov)
            # success => return
            return cov
        except RuntimeError:
            # eigen-decompose, clamp eigenvalues to >= eps, rebuild
            vals, vecs = torch.linalg.eigh(cov)
            vals_clamped = torch.clamp(vals, min=eps)
            cov_pd = (vecs @ torch.diag(vals_clamped) @ vecs.T)
            # renormalize small numeric asymmetry and return
            cov_pd = 0.5 * (cov_pd + cov_pd.T)
            return cov_pd

    def _multivariate_gaussian_pdf(self, X: torch.Tensor, mean: torch.Tensor, 
                                 cov_inv: torch.Tensor, cov_det: torch.Tensor) -> torch.Tensor:
        """
        Compute multivariate Gaussian PDF for a single blob.
        """
        d = self.dim
        X_centered = X - mean
        
        # Compute quadratic form: (x-μ)^T Σ^{-1} (x-μ)
        if X.dim() == 1:
            quad_form = X_centered @ cov_inv @ X_centered
        else:
            quad_form = torch.einsum('ni,ij,nj->n', X_centered, cov_inv, X_centered)
        
        # Multivariate Gaussian PDF formula
        normalization = 1.0 / torch.sqrt((2 * math.pi) ** d * cov_det)
        pdf_value = normalization * torch.exp(-0.5 * quad_form)
        
        return pdf_value

    def _evaluate_single_point(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the function at single or multiple points.
        """
        result = torch.zeros(X.shape[0])
        
        # Sum contributions from all blobs
        for i in range(self.n_blobs):
            pdf_val = self._multivariate_gaussian_pdf(
                X, self.means[i], self.cov_invs[i], self.cov_dets[i]
            )
            result += self.weights[i] * pdf_val
        
        return result.unsqueeze(-1)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return self._evaluate_single_point(X).reshape(-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y

    def get_blob_info(self) -> dict:
        """Return information about the blobs for analysis."""
        return {
            'means': self.means,
            'weights': self.weights,
            'rhos': self.rhos,
            'optimal_point': self.optimal_point,
            'optimal_value': self.optimal_value
        }    

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
        normalize_search: whether to run numeric search to estimate extrema (default True)
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
        seed: Optional[int] = None,
        rho_max: float = 0.4,
        normalize_method: Literal["max", "minmax"] = "max",
        normalize_search: bool = True,
        n_random_samples: int = 5000,
        refine_restarts: int = 8,
        refine_steps: int = 200,
        refine_lr: float = 0.05,
        jitter: float = 1e-8,
    ):
        # --- attributes BaseTestProblem expects BEFORE super().__init__() ---
        self.d = int(d)
        self.dim = self.d
        # treat all inputs as continuous
        self.continuous_inds = list(range(self.d))
        self.discrete_inds = []
        self.categorical_inds = []
        # bounds on original domain
        self._bounds = [(0.0, 10.0) for _ in range(self.d)]

        # initialize BaseTestProblem / nn.Module internals
        super().__init__()

        # basic attributes
        self.noise_std = float(noise_std)
        self.negate = bool(negate)

        # set number of blobs
        self.n_blobs = n_blobs if n_blobs is not None else self.d

        # RNG
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # storage lists (will convert to tensors)
        means_list = []
        cov_list = []
        weights_list = []
        rhos_list = []

        dtype = torch.get_default_dtype()
        device = torch.device("cpu")

        # safe rho sampling range for exchangeable corr
        rho_upper = float(rho_max)
        rho_lower_theory = -1.0 / float(self.d - 1)
        eps = 1e-8
        rho_lower = rho_lower_theory + 1e-6
        if rho_lower >= rho_upper:
            rho_lower = rho_upper - 1e-6

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
            # add tiny jitter for numeric stability
            Cj = C + torch.eye(self.d, dtype=C.dtype, device=C.device) * (1e-10)
            # Cholesky (should succeed as we regularized earlier)
            try:
                L = torch.linalg.cholesky(Cj)
            except RuntimeError:
                # last resort: eigen-clamp
                vals, vecs = torch.linalg.eigh(Cj)
                vals_clamped = torch.clamp(vals, min=1e-12)
                Cj = vecs @ torch.diag(vals_clamped) @ vecs.T
                Cj = 0.5 * (Cj + Cj.T)
                L = torch.linalg.cholesky(Cj)

            # compute inverse via cholesky_inverse for stability
            # torch.cholesky_inverse expects Cholesky factor from torch.cholesky; for modern torch use:
            invC = torch.cholesky_inverse(L) if hasattr(torch, "cholesky_inverse") else torch.linalg.inv(Cj)
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
        self._normalize_method = normalize_method
        self._scale = max(float(analytic_val), 1e-12)  # fallback scale
        self._min_obs = 0.0
        self._max_obs = float(analytic_val)

        if normalize_search:
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

            # optionally refine best point via gradient ascent (only for 'max' mode)
            if normalize_method == "max":
                # find a reasonable starting candidate: analytic mean and top random sample
                # We'll pick the best-of-k random candidates for gradient starts
                # Small set of restarts for refinement
                best_ref_val = best_val
                best_ref_x = self.optimal_point.clone()
                # seed starts: use a few random points plus analytic mean
                starts = [self.optimal_point.clone()]
                for _ in range(refine_restarts - 1):
                    starts.append(10.0 * torch.rand(self.d, dtype=dtype, device=device))

                for s in starts:
                    x = s.clone().detach().to(dtype).requires_grad_(True)
                    opt = torch.optim.Adam([x], lr=refine_lr)
                    for _ in range(refine_steps):
                        opt.zero_grad()
                        v = self._evaluate_single_point(x.unsqueeze(0)).squeeze()
                        # maximize -> minimize negative
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
            else:
                # 'minmax' mode: set observed range from sampling
                # ensure sensible defaults if sampling was too small
                if worst_val == float("inf"):
                    worst_val = 0.0
                self._min_obs = float(min(worst_val, 0.0))
                self._max_obs = float(max(best_val, analytic_val, 1e-12))
                # keep scale for backward compatibility, but main scheme will be min-max
                self._scale = float(max(self._max_obs, 1e-12))

        # finalize normalized optimal_value
        if self._normalize_method == "max":
            self.optimal_value = 1.0
        else:
            # minmax mapping yields observed max -> 1.0
            self.optimal_value = 1.0

    # ----------------- helpers -----------------
    def _create_exchangeable_covariance(self, stds: torch.Tensor, rho: float, eps: float = 1e-8) -> torch.Tensor:
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

        # quick PD check with cholesky
        try:
            torch.linalg.cholesky(cov)
            return cov
        except RuntimeError:
            vals, vecs = torch.linalg.eigh(cov)
            vals_clamped = torch.clamp(vals, min=eps)
            cov_pd = vecs @ torch.diag(vals_clamped) @ vecs.T
            cov_pd = 0.5 * (cov_pd + cov_pd.T)
            # ensure tiny jitter so later cholesky succeeds
            cov_pd = cov_pd + torch.eye(d, dtype=cov_pd.dtype, device=cov_pd.device) * (1e-12)
            return cov_pd

    def _multivariate_gaussian_pdf(self, X: torch.Tensor, mean: torch.Tensor, cov_inv: torch.Tensor, cov_logdet: float) -> torch.Tensor:
        """
        Evaluate multivariate Gaussian pdf at rows of X for given mean, precision (cov_inv) and logdet.
        Returns (N,) for X shape (N,d) or scalar for (d,) inputs.
        """
        if X.dim() == 1:
            Xc = X - mean
            quad = float((Xc @ cov_inv @ Xc).item())
            norm_log = -0.5 * (self.d * math.log(2.0 * math.pi) + float(cov_logdet))
            return torch.tensor(math.exp(norm_log - 0.5 * quad), dtype=mean.dtype)
        else:
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
        if self._normalize_method == "max":
            scaled = vals / float(max(self._scale, 1e-12))
            scaled = torch.clamp(scaled, min=0.0, max=1.0)
            return scaled.reshape(-1)
        else:  # minmax
            denom = float(self._max_obs - self._min_obs) if (self._max_obs - self._min_obs) > 1e-12 else 1.0
            scaled = (vals - float(self._min_obs)) / denom
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

class GoldsteinPrice(BaseTestProblem):
    """
    Goldstein–Price function in standard minimization form:

      f(x1, x2) =
        [1 + (x1 + x2 + 1)^2 * (19 - 14x1 + 3x1^2 - 14x2 + 6x1x2 + 3x2^2)] *
        [30 + (2x1 - 3x2)^2 * (18 - 32x1 + 12x1^2 + 48x2 - 36x1x2 + 27x2^2)]

    - Domain: x1, x2 ∈ [-2, 2]
    - Global MINIMUM value: 3.0 at (0, -1)
    """

    def __init__(self, noise_std: float = 0.0, negate: bool = False):
        self.dim = 2
        self._bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        self.noise_std = noise_std
        self.negate = negate
        self.optimal_value = 3.0  # known global minimum

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        x1, x2 = X[..., 0], X[..., 1]

        term1 = (
            1
            + (x1 + x2 + 1) ** 2
            * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
        )
        term2 = (
            30
            + (2 * x1 - 3 * x2) ** 2
            * (
                18
                - 32 * x1
                + 12 * x1**2
                + 48 * x2
                - 36 * x1 * x2
                + 27 * x2**2
            )
        )
        y = term1 * term2
        return y.reshape(-1)  # ensure output shape (N, 1)
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y
 
class RosenbrockRotated(BaseTestProblem):
    """
    Rosenbrock function with rotated cross-terms for non-additive decomposition.
    
    f(x) = Σ[100*(x_{i+1} - x_i²)² + (1-x_i)²] applied to rotated coordinates
    
    - Domain: x_i ∈ [-5, 5] (extended from standard Rosenbrock bounds)
    - Global MINIMUM value: 0.0 at x = [1, 1, ..., 1] (before rotation)
    - After rotation, minimum shifts but value remains 0.0
    """
    
    def __init__(self, dim: int = 5, noise_std: float = 0.0, negate: bool = False, 
                 rotation_seed: Optional[int] = None):
        self.dim = dim
        self._bounds = [(-5.0, 5.0) for _ in range(dim)]
        self.noise_std = noise_std
        self.negate = negate
        self.optimal_value = 0.0  # Known global minimum value
        
        # Generate random rotation matrix
        if rotation_seed is not None:
            torch.manual_seed(rotation_seed)
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        self.rotation_matrix = Q
        
        # Precompute the inverse rotation to find optimal point in rotated space
        # The optimum in original space is at [1, 1, ..., 1]
        original_optimum = torch.ones(dim)
        self.rotated_optimum = self.rotation_matrix @ original_optimum

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # Apply rotation to input
        X_rotated = X @ self.rotation_matrix.T
        
        # Standard Rosenbrock function on rotated coordinates
        result = torch.zeros(X.shape[0])
        for i in range(self.dim - 1):
            term1 = 100 * (X_rotated[..., i+1] - X_rotated[..., i]**2)**2
            term2 = (1 - X_rotated[..., i])**2
            result += term1 + term2
            
        return result.reshape(-1)  # ensure output shape (N, 1)
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y

class AckleyCorrelated(BaseTestProblem):
    """
    Ackley function with dimension correlations for non-additive decomposition.
    
    Modified Ackley with neighbor correlations in the squared term:
    f(x) = -a*exp(-b*sqrt(Σ(x_i + α*x_{i-1} + α*x_{i+1})²/n)) - exp(Σcos(c*x_i)/n) + a + exp(1)
    
    - Domain: x_i ∈ [-32.768, 32.768] (standard Ackley bounds)
    - Global MINIMUM value: 0.0 at x = [0, 0, ..., 0]
    - Correlation changes landscape but preserves minimum at origin
    """
    
    def __init__(self, dim: int = 5, noise_std: float = 0.0, negate: bool = False, 
                 correlation_strength: float = 0.15):
        self.dim = dim
        self._bounds = [(-32.768, 32.768) for _ in range(dim)]
        self.noise_std = noise_std
        self.negate = negate
        self.optimal_value = 0.0
        self.correlation_strength = correlation_strength
        self.a = 20.0
        self.b = 0.2
        self.c = 2.0 * torch.pi

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        n = self.dim
        a, b, c = self.a, self.b, self.c
        alpha = self.correlation_strength
        
        # Compute correlated squared sum with neighborhood interactions
        correlated_sum = torch.zeros(X.shape[0])
        for i in range(n):
            neighbor_contrib = torch.zeros(X.shape[0])
            if i > 0:
                neighbor_contrib += alpha * X[..., i-1]
            if i < n-1:
                neighbor_contrib += alpha * X[..., i+1]
            correlated_sum += (X[..., i] + neighbor_contrib)**2
        
        term1 = -a * torch.exp(-b * torch.sqrt(correlated_sum / n))
        term2 = -torch.exp(torch.sum(torch.cos(c * X), dim=-1) / n)
        result = term1 + term2 + a + torch.exp(torch.tensor(1.0))
        
        return result.reshape(-1)  # ensure output shape (N, 1)
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y
  
class CyclicalFunction(BaseTestProblem):
    """
    Cyclical function:
      f(x) = sum_{i=0}^{d-1} x_{(i+1) mod d} * exp(cos(x_i))

    Args:
        d: input dimension
        noise_std: observation noise
        negate: whether to negate outputs
        seed: optional RNG seed
    """

    def __init__(self, d: int, noise_std: 0.0, negate: bool = False, seed: Optional[int] = None,):
        
        self.d = int(d)
        self.dim = self.d
        self._bounds = [(0.0, 10.0) for _ in range(self.d)]

        # mark all inputs continuous (common for these synthetic problems)
        self.continuous_inds = list(range(self.d))
        self.discrete_inds = []
        self.categorical_inds = []

        # initialize Module / BaseTestProblem internals
        super().__init__()

        # now safe to attach other attributes
        if seed is not None:
            torch.manual_seed(seed)

        self.noise_std = float(noise_std)
        self.negate = bool(negate)

        self.optimal_value = 17.29225 * float(self.d)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)
    
    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Vectorized evaluation:
          For batch X shape (N,d) -> returns (N,)
          For single vector X shape (d,) -> returns (1,) (reshapeable to scalar)
        Computes sum_i x_{i+1} * exp(cos(x_i)) with cyclic indexing.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        # ensure batch dimension
        squeezed = False
        if X.dim() == 1:
            X = X.unsqueeze(0)
            squeezed = True

        # X is (N, d)
        if X.shape[1] != self.d:
            raise ValueError(f"Input last-dimension must be {self.d}, got {X.shape[1]}")

        # x_{i+1} with wrap-around: roll left by -1
        X_next = torch.roll(X, shifts=-1, dims=1)  # (N, d)

        # compute exp(cos(x_i)) elementwise
        trig_factor = torch.exp(torch.cos(X))      # (N, d)

        # element-wise product and sum across coordinates
        terms = X_next * trig_factor               # (N, d)
        fvals = torch.sum(terms, dim=1)            # (N,)

        # return 1-D tensor of length N (if original input was 1-D, return a length-1 tensor)
        return fvals.reshape(-1)

    def forward(self, X):
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y
    


### Functions that need re-design


class GriewankRosenbrockHybrid(BaseTestProblem):
    """
    Hybrid function with first half dimensions as Ackley and second half as Rosenbrock.
    
    For x = [x_ackley, x_rosenbrock] where:
    - x_ackley: first ceil(d/2) dimensions evaluated with Ackley function
    - x_rosenbrock: remaining dimensions evaluated with Rosenbrock function
    
    f(x) = Ackley(x[0:ceil(d/2)]) + Rosenbrock(x[ceil(d/2):])
    
    - Domain: x_i ∈ [-5, 5] (works well for both functions)
    - Global MINIMUM value: 0.0 at x_ackley = [0,0,...,0] and x_rosenbrock = [1,1,...,1]
    """

    def __init__(self, dim: int = 5, noise_std: float = 0.0, negate: bool = False):
        self.dim = dim
        self._bounds = [(-5.0, 5.0) for _ in range(dim)]
        self.noise_std = noise_std
        self.negate = negate
        
        # Split dimensions: first half Ackley, second half Rosenbrock
        self.ackley_dim = (dim + 1) // 2  # ceil(d/2)
        self.rosenbrock_dim = dim - self.ackley_dim
        
        # Optimal point: zeros for Ackley part, ones for Rosenbrock part
        self.optimal_point = torch.cat([
            torch.zeros(self.ackley_dim),  # Ackley optimum at [0,0,...,0]
            torch.ones(self.rosenbrock_dim)  # Rosenbrock optimum at [1,1,...,1]
        ])
        
        # Calculate optimal value by evaluating both parts at their optima
        ackley_opt = self._ackley_part(self.optimal_point[:self.ackley_dim].unsqueeze(0))
        rosenbrock_opt = self._rosenbrock_part(self.optimal_point[self.ackley_dim:].unsqueeze(0))
        self.optimal_value = (ackley_opt + rosenbrock_opt).item()

    def _ackley_part(self, x_ackley: torch.Tensor) -> torch.Tensor:
        """Ackley function applied to the first half of dimensions."""
        if self.ackley_dim == 0:
            return torch.zeros(x_ackley.shape[0], 1)
            
        a, b, c = 20.0, 0.2, 2.0 * torch.pi
        n = self.ackley_dim
        
        sum_sq = torch.sum(x_ackley**2, dim=-1)
        sum_cos = torch.sum(torch.cos(c * x_ackley), dim=-1)
        
        term1 = -a * torch.exp(-b * torch.sqrt(sum_sq / n))
        term2 = -torch.exp(sum_cos / n)
        
        return term1 + term2 + a + torch.exp(torch.tensor(1.0)).unsqueeze(-1)

    def _rosenbrock_part(self, x_rosenbrock: torch.Tensor) -> torch.Tensor:
        """Rosenbrock function applied to the second half of dimensions."""
        if self.rosenbrock_dim == 0:
            return torch.zeros(x_rosenbrock.shape[0], 1)
        elif self.rosenbrock_dim == 1:
            # For single dimension Rosenbrock, use (1-x)^2
            return ((1 - x_rosenbrock[..., 0])**2).unsqueeze(-1)
            
        result = torch.zeros(x_rosenbrock.shape[0])
        for i in range(self.rosenbrock_dim - 1):
            term1 = 100 * (x_rosenbrock[..., i+1] - x_rosenbrock[..., i]**2)**2
            term2 = (1 - x_rosenbrock[..., i])**2
            result += term1 + term2
            
        return result.unsqueeze(-1)
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # Split input into Ackley and Rosenbrock parts
        x_ackley = X[..., :self.ackley_dim]
        x_rosenbrock = X[..., self.ackley_dim:]
        
        # Evaluate each part separately
        ackley_val = self._ackley_part(x_ackley)
        rosenbrock_val = self._rosenbrock_part(x_rosenbrock)
        
        # Combine results
        return ackley_val.reshape(-1) + rosenbrock_val.reshape(-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y


class MultiplicativeInteraction(BaseTestProblem):
    """
    Multiplicative coupling of two BoTorch synthetic subfunctions U and V on disjoint subsets.
    f(x) = U(x_S1) * V(x_S2) + alpha * (U(x_S1) + V(x_S2))

    Key features:
      - U and V use the same per-dimension bounds (default (0,10)).
      - default u_kind='ackley', v_kind='rosenbrock'
      - computes self.optimal_x and self.optimal_value at construction (approx. if needed).
    """

    def __init__(self, d=6, noise_std=0.0, negate=False, split='half',
                 u_kind='ackley', v_kind='rosenbrock', alpha=0.5,
                 rescale_positive=True, seed: int | None = None,
                 # optimization knobs for computing optimum:
                 sub_n_restarts: int = 12, sub_steps: int = 300, sub_lr: float = 0.05,
                 full_n_restarts: int = 20, full_steps: int = 400, full_lr: float = 0.05):
        
        self.d = d
        self.dim = d
         # Mark all input indices as continuous (common for synthetic, real-valued benchmarks)
        self.continuous_inds = list(range(self.d))
        self.discrete_inds = []
        self.categorical_inds = []
        self._bounds = [(0.0, 10.0)] * self.d
        
        
        super().__init__() # IMPORTANT: initialize parent Module before attaching any submodules

        
        self.noise_std = float(noise_std)
        self.negate = negate
        self.alpha = float(alpha)
        self.rescale_positive = bool(rescale_positive)

        dtype = torch.get_default_dtype()
        if seed is not None:
            torch.manual_seed(seed)

        # determine split
        if split == 'half':
            k = self.d // 2
        elif isinstance(split, int):
            k = int(split)
            if k <= 0 or k >= self.d:
                raise ValueError("split integer must satisfy 0 < k < d")
        else:
            raise ValueError("split must be 'half' or an integer")

        self.idx_u = list(range(0, k))
        self.idx_v = list(range(k, self.d))
        if len(self.idx_u) == 0 or len(self.idx_v) == 0:
            raise ValueError("split produced empty group")

        # Sub-bounds: equal per-dimension bounds for both subproblems (same domain specification)
        # i.e., each subproblem gets [(0,10)] repeated for its sub-dim
        bounds_sub_u = [(0.0, 10.0)] * len(self.idx_u)
        bounds_sub_v = [(0.0, 10.0)] * len(self.idx_v)

        # instantiate BoTorch synthetic subproblems
        def make_botorch(kind, dim):
            k = kind.lower()
            if k == 'ackley':
                return Ackley(dim, noise_std=0.0, negate=self.negate)
            elif k == 'griewank':
                return Griewank(dim, noise_std=0.0, negate=self.negate)
            elif k == 'rosenbrock' or k == 'rosen':
                # Rosenbrock in botorch is defined for arbitrary dim
                return Rosenbrock(dim, noise_std=0.0, negate=self.negate)
            else:
                raise ValueError(f"unsupported subfunction kind '{kind}'")

        self.U = make_botorch(u_kind, len(self.idx_u))
        self.V = make_botorch(v_kind, len(self.idx_v))

        # store optimization knobs
        self._sub_n_restarts = int(sub_n_restarts)
        self._sub_steps = int(sub_steps)
        self._sub_lr = float(sub_lr)
        self._full_n_restarts = int(full_n_restarts)
        self._full_steps = int(full_steps)
        self._full_lr = float(full_lr)

        # compute optimal
        try:
            self.optimal_x, self.optimal_value = self._compute_optimal()
        except Exception as e:
            # if anything fails, set None and rethrow optional debug info - but we prefer to keep class usable
            self.optimal_x = None
            self.optimal_value = None
            # re-raise so caller is aware (or comment this line to swallow errors)
            raise

    # -----------------------------
    # utility: projection/clamp to bounds
    def _clamp_to_bounds(self, x):
        # x: (d,) or (N,d) tensor
        lb = torch.tensor([b[0] for b in self._bounds], dtype=x.dtype, device=x.device)
        ub = torch.tensor([b[1] for b in self._bounds], dtype=x.dtype, device=x.device)
        return torch.max(torch.min(x, ub), lb)

    def _clamp_to_sub_bounds(self, x_sub, which='u'):
        # x_sub: (m,) or (N,m); which in {'u','v'}
        if which == 'u':
            m = len(self.idx_u)
        else:
            m = len(self.idx_v)
        lb = torch.tensor([0.0] * m, dtype=x_sub.dtype, device=x_sub.device)
        ub = torch.tensor([10.0] * m, dtype=x_sub.dtype, device=x_sub.device)
        return torch.max(torch.min(x_sub, ub), lb)

    # -----------------------------
    # maximize a subfunction via multi-start projected gradient ascent.
    # func_eval: callable that accepts X_sub (N,m) and returns (N,) tensor values to maximize.
    def _maximize_sub(self, func_eval, m, n_restarts, steps, lr):
        dtype = torch.get_default_dtype()
        device = torch.device('cpu')
        best_val = -float("inf")
        best_x = None

        # starts: include center (middle of bounds) and some random starts
        center = 5.0 * torch.ones(m, dtype=dtype, device=device)
        starts = [center.clone()]
        n_rand = max(0, n_restarts - len(starts))
        for _ in range(n_rand):
            starts.append((0.0 + (10.0 - 0.0) * torch.rand(m, dtype=dtype, device=device)))

        for s in starts:
            x = s.clone().detach().to(dtype).requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                val = func_eval(x.unsqueeze(0)).squeeze()  # scalar tensor
                # we will maximize -> minimize negative
                loss = -val
                (loss).backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(0.0, 10.0)

            final_val = float(func_eval(x.unsqueeze(0)).squeeze().item())
            if final_val > best_val:
                best_val = final_val
                best_x = x.detach().clone()

        return best_x, best_val

    # -----------------------------
    # evaluate multiplicative mixture given full X (N,d) or (d,)
    def _evaluate_true(self, X):
        is_vector = (X.dim() == 1)
        if is_vector:
            X = X.unsqueeze(0)

        Xu = X[:, self.idx_u]
        Xv = X[:, self.idx_v]

        u_raw = self.U.evaluate_true(Xu)
        v_raw = self.V.evaluate_true(Xv)

        if self.rescale_positive:
            u = torch.exp(-u_raw / (1.0 + Xu.shape[-1]))
            v = torch.exp(-v_raw / (1.0 + Xv.shape[-1]))
        else:
            u = u_raw
            v = v_raw

        f = u * v + self.alpha * (u + v)
        return f.reshape(-1)  # ensure output shape (N, 1)

    # -----------------------------
    # helper to compute per-subfunction maxima and fallback to full-space search if needed
    def _compute_optimal(self):
        dtype = torch.get_default_dtype()
        device = torch.device('cpu')

        # Define sub eval functions depending on rescale_positive
        if self.rescale_positive:
            def eval_u_pos(Xsub):
                # Xsub: (N,m)
                val = self.U.evaluate_true(Xsub)
                return torch.exp(-val / (1.0 + Xsub.shape[-1]))
            def eval_v_pos(Xsub):
                val = self.V.evaluate_true(Xsub)
                return torch.exp(-val / (1.0 + Xsub.shape[-1]))
            u_eval = eval_u_pos
            v_eval = eval_v_pos
        else:
            u_eval = lambda Xsub: self.U.evaluate_true(Xsub)
            v_eval = lambda Xsub: self.V.evaluate_true(Xsub)

        # maximize U on its subspace
        m_u = len(self.idx_u)
        m_v = len(self.idx_v)
        x_u_star, u_max = self._maximize_sub(u_eval, m_u,
                                            n_restarts=self._sub_n_restarts,
                                            steps=self._sub_steps, lr=self._sub_lr)
        x_v_star, v_max = self._maximize_sub(v_eval, m_v,
                                            n_restarts=self._sub_n_restarts,
                                            steps=self._sub_steps, lr=self._sub_lr)

        # If both succeeded (not None) and we used positive rescaling, the product is monotone:
        if (x_u_star is not None) and (x_v_star is not None) and self.rescale_positive:
            # best full-x is the concatenation
            full_x = torch.zeros(self.d, dtype=dtype, device=device)
            full_x[self.idx_u] = x_u_star
            full_x[self.idx_v] = x_v_star
            # compute mixture value exactly
            f_val = float((u_max * v_max + self.alpha * (u_max + v_max)))
            return full_x, f_val

        # Otherwise, fallback to a full-space multi-start gradient ascent.
        # Build start points seeded from (x_u_star concat x_v_star) plus random restarts.
        starts = []
        if x_u_star is not None and x_v_star is not None:
            seed_x = torch.zeros(self.d, dtype=dtype, device=device)
            seed_x[self.idx_u] = x_u_star
            seed_x[self.idx_v] = x_v_star
            starts.append(seed_x)
        # also add starts at all-subspace centers and some random starts
        starts.append(5.0 * torch.ones(self.d, dtype=dtype, device=device))
        for _ in range(max(0, self._full_n_restarts - len(starts))):
            starts.append((0.0 + (10.0 - 0.0) * torch.rand(self.d, dtype=dtype, device=device)))

        best_val = -float("inf")
        best_x = None

        for s in starts:
            x = s.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=self._full_lr)
            for _ in range(self._full_steps):
                opt.zero_grad()
                val = self._evaluate_true(x.unsqueeze(0)).squeeze()
                loss = -val
                loss.backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(0.0, 10.0)

            val_end = float(self._evaluate_true(x.unsqueeze(0)).squeeze().item())
            if val_end > best_val:
                best_val = val_end
                best_x = x.detach().clone()

        # final safety: also check all combinations of top subspace points if available:
        # (if we had only x_u_star or only x_v_star, still helpful)
        if x_u_star is not None and x_v_star is not None:
            # compute also the mixture at means of U/V (or at found x_u_star/x_v_star)
            combo = torch.zeros(self.d, dtype=dtype, device=device)
            combo[self.idx_u] = x_u_star
            combo[self.idx_v] = x_v_star
            val_combo = float(self._evaluate_true(combo.unsqueeze(0)).squeeze().item())
            if val_combo > best_val:
                best_val = val_combo
                best_x = combo.detach().clone()

        return best_x, best_val
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        # compatibility wrapper
        return self._evaluate_true(X)
    # -----------------------------
    def forward(self, X):
        Y = self._evaluate_true(X)
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
        return -Y if self.negate else Y
     