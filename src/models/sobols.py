import numpy as np
import torch
import copy

from SALib.sample import sobol as sobol_sampler
from scipy.stats import sobol_indices as scipy_sobol
from scipy.stats import qmc, uniform
import tntorch as tn
import gpytorch

def optimize(gp, train_x, train_y, n_iter=20, lr=0.01):
    """
    Train an GP + Likelihood model.

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


class Sobol:
    """
    Uses SALib to compute variance-based Sobol indices (including second-order),
    and produce a partition based on thresholds of interaction strength.

    Attributes:
      epsilon : float threshold for 'additivity' interaction
      problem : dict for SALib problem definition (names, bounds, etc.)
      method: string describing what Sobol global sensitivity analysis method to use.
      M: int (power of two) for sobol sampler
      B: int bootstrap for sobol methods

    Methods:
      compute_interactions(train_x, train_y) -> interactions matrix (d x d numpy)
      update_partition(interactions) -> partition (list of list of dims)
    """

    def __init__(self, f_obj, epsilon=8e-2, method='scipy', M=2048, B=128):
        """
        f_obj: SyntheticTestFun object for the test function to optimize
        epsilon: threshold for high-order sobol interactions
        B: number of bootstrap samples
        M: number of monte-carlo samples for sobol metamodel
        method: string representing the Sobol global sensitivity analysis method to use.
        """
        self.epsilon = 0.08 # - 0.02 * min(1.0, (d**2 / 30.0))
        self.B = B
        self.M = M
        self.problem = self._build_problem(f_obj)
        self.device = torch.device("cpu")

        method_map = {
            'scipy': self.interactions_scipy,
            'wirthl': self.interactions_wirthl,
            'deriv': self.interactions_deriv,
            'tt': self.interactions_tt,
            'asm': self.interactions_asm
        }
        self.method = method_map[method]


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
    
    def to(self, device):
        """
        Sets the device for the Sobol analyzer and returns self.
        This allows chaining: sobol = Sobol(f).to(device)
        """
        self.device = torch.device(device)
        return self

    def interactions_wirthl(self, train_x, train_y, metamodel):
        """
        Calculate higher-order Sobol indices using GP metamodel as described in the paper
        Global Sensitivity Analysis based on GP metamodelling for complex biomechanical problems.
        
        Args:
            train_x: Tensor (n, d) or array - training inputs
            train_y: Tensor or array (n,) or (n,1) - training outputs  
            simulator: trained ExactGP model
            likelihood: corresponding GPyTorch likelihood            
        Returns:
            high-order interactions mean, variance and 95% confidence interval
        """

        #problem vars
        d = self.problem['num_vars']
        self.NZ = train_x.shape[0] 
        
        # tensor converters
        device = train_x.device
        dtype = train_x.dtype
        
        # Train the meta model
        metamodel, metamodel.likelihood, _ = optimize(metamodel, train_x, train_y)

        # Generate Monte-Carlo samples
        A = sobol_sampler.sample(self.problem, self.M, calc_second_order=True, skip_values=self.M*2)
        B = sobol_sampler.sample(self.problem, self.M, calc_second_order=True, skip_values=self.M*2)
        A, B = torch.tensor(A, device=device, dtype=dtype), torch.tensor(B, device=device, dtype=dtype)
        A_B = {i: copy.deepcopy(A) for i in range(d)}
        B_A = {i: copy.deepcopy(B) for i in range(d)}
        for i in range(d):
            A_B[i][:, i] = B[:, i]
            B_A[i][:, i] = A[:,i] 

        high_order_estimates = torch.zeros((self.NZ, self.B, d))
        second_order_estimates = torch.zeros((self.NZ, self.B, d, d))

        metamodel.eval(); metamodel.likelihood.eval()

        # Repeat for all metamodel realizations
        for k in range(self.NZ):

            # Sample realisations of metamodel at MC samples
            with torch.no_grad():

                # Get metamodel realization
                f_A_post = metamodel.likelihood(metamodel(A))
                f_B_post = metamodel.likelihood(metamodel(B))
                f_A_B_post = {i: metamodel.likelihood(metamodel(A_B[i])) for i in range(d)}
                f_B_A_post = {i: metamodel.likelihood(metamodel(B_A[i])) for i in range(d)}

                for b in range(self.B):

                    # Get bootstrap sample
                    f_A_sample = f_A_post.sample(sample_shape=torch.Size([self.B]))
                    f_B_sample = f_B_post.sample(sample_shape=torch.Size([self.B]))
                    f_A_B_sample = {i: f_A_B_post[i].sample(sample_shape=torch.Size([self.B])) for i in range(d)}
                    f_B_A_sample = {i: f_B_A_post[i].sample(sample_shape=torch.Size([self.B])) for i in range(d)}

                    # Valculate total variance between A, B
                    V = torch.var(torch.cat([f_A_sample.flatten(), f_B_sample.flatten()]))
                    V += 1e-9

                    #Saltelli formula for first order interaction
                    S_1 = torch.zeros(d)
                    for i in range(d):
                        numerator = torch.mean(f_B_sample*(f_A_B_sample[i] - f_A_sample))
                        S_1[i] = numerator / V

                    # Jansen formula for total order interaction
                    S_T = torch.zeros(d)
                    for i in range(d):
                        numerator = 0.5 * torch.mean((f_A_sample - f_A_B_sample[i])**2)
                        S_T[i] = numerator / V

                    # Calculate second order interactions
                    S_2 = torch.ones((d, d))
    
                    for i in range(d):
                        for j in range(i+1, d):
                            numerator = torch.mean((f_A_B_sample[i]*f_B_A_sample[j]) - (f_A_sample*f_B_sample))
                            S_2[i, j] = (numerator / V) - S_1[i] - S_1[j]
                            S_2[j, i] = S_2[i, j]

                    # Calculate high-order indices for bootstrap, gp_realization sample
                    high_order_estimates[k, b, : ] = S_T - S_1
                    second_order_estimates[k, b, :, :] = S_2
        
        # Evaluate statistics: mean + variance
        high_order_interactions = high_order_estimates.reshape(-1, d).cpu().numpy()
        second_order_interactions = second_order_estimates.cpu().numpy()

        # (Equation 19)
        high_order_mean = np.mean(high_order_interactions, axis=0)
        # (Equation 20)
        high_order_variance = np.var(high_order_interactions, axis=0, ddof=1)

        # Calculate confidence intervals (95% by default)
        confidence_level = 0.95
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = np.column_stack([
            np.percentile(high_order_interactions, lower_percentile, axis=0),
            np.percentile(high_order_interactions, upper_percentile, axis=0)
        ])

        print(f'Results for interactions of Wirthl (SobolGP) method\n')
        print(f'Higher order interactions: {high_order_mean} with var: {high_order_variance}')

        #print(f'mean second order interactions: \n {np.mean(second_order_interactions, axis=(0,1))}')

        #return high_order_mean, high_order_variance, confidence_intervals        

    def interactions_scipy(self, train_x, train_y, metamodel):
        """
        Calculate higher-order Sobol indices using GP metamodel and Scipy's machinery 
        
        Args:
            train_x: Tensor (n, d) or array - training inputs
            train_y: Tensor or array (n,) or (n,1) - training outputs  
            simulator: trained ExactGP model
            likelihood: corresponding GPyTorch likelihood            
        Returns:
            high-order interactions mean, variance and 95% confidence interval
        """

        #problem vars
        d = self.problem['num_vars']
        self.NZ = train_x.shape[0]
        
        # tensor converters
        device = train_x.device
        dtype = train_x.dtype
        
        # Train the meta model
        metamodel, metamodel.likelihood, _ = optimize(metamodel, train_x, train_y)

        # Define Distributions for Scipy
        bounds = np.array(self.problem['bounds'], dtype=np.float32)
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        
        dists = [
            uniform(loc=b[0], scale=b[1] - b[0]) 
            for b in bounds
        ]

        metamodel.eval(); metamodel.likelihood.eval()
        # Define Wrapper Function
        def gp_mean_wrapper(x_np):
            
            x_tens = torch.tensor(np.transpose(x_np), device=device, dtype=dtype) # Convert Scipy's numpy input to Torch tensor
            with torch.no_grad():
                output = metamodel.likelihood(metamodel(x_tens))
                mean_pred = output.mean
            return mean_pred.cpu().numpy().reshape(-1) # Return numpy array to Scipy


        # ---- Bootstrap storage ----
        S1_boot = np.zeros((self.B, d))
        ST_boot = np.zeros((self.B, d))
        high_boot = np.zeros((self.B, d))


        for b in range(self.B):

            result = scipy_sobol(
            func=gp_mean_wrapper, 
            n=self.B, 
            dists=dists, 
            )

            # Extract components
            S1 = result.first_order     # (d,)
            ST = result.total_order     # (d,)
            # S2 = res["S2"]    # (d, d) -- available if needed

            #print(f'confidence interval: {result.bootstrap().total_order.confidence_interval}')

            S1_boot[b] = S1
            ST_boot[b] = ST
            high_boot[b] = ST - S1     # higher-order index


        # ---- Aggregation over bootstrap ----
        high_mean = high_boot.mean(axis=0)
        high_var = high_boot.var(axis=0, ddof=1)
        S1_tot = S1_boot.mean(axis=0)
        ST_tot = ST_boot.mean(axis=0)

        #print(f'predicted S1: {S1_tot} and predicted S_total: {ST_tot}')

        # 95% CI
        lower = np.percentile(high_boot, 2.5, axis=0)
        upper = np.percentile(high_boot, 97.5, axis=0)
        CI = np.vstack([lower, upper]).T

        #print(f'Results for interactions of Saltelli-scipy (SobolGP) method\n')
        #print(f'Higher order interactions: {high_mean} with var: {high_var}')

        return high_mean

    def interactions_deriv(self, train_x, train_y, metamodel):
        """
        Calculate derivative-based sensitivity indices (DGSM) using GP gradients.
        Serves as a computationally efficient proxy for Total Interaction indices.

        Approximation:
            S_tot_i ~ E[(df/dx_i)^2] * Var(X_i) / Var(Y)
        
        Args:
            train_x: Tensor (n, d)
            train_y: Tensor (n, 1)
            metamodel: GPyTorch model
            
        Returns:
            dgsm_mean: (d,) array of normalized derivative indices
            dgsm_var: (d,) variance of the index (across GP realizations)
            confidence_intervals: (d, 2) 95% CI
        """
        
        # 1. Setup & Optimization
        d = self.problem['num_vars']
        self.NZ = train_x.shape[0] # Keeping your convention of using N for realizations
        
        device = train_x.device
        dtype = train_x.dtype

        # Train/Optimize the metamodel
        metamodel, metamodel.likelihood, _ = optimize(metamodel, train_x, train_y)
        metamodel.eval(); metamodel.likelihood.eval()

        # 2. Generate Evaluation Points (Sobol sequence, but just N samples)
        X_eval = torch.tensor(sobol_sampler.sample(self.problem, self.M, calc_second_order=False), device=device, dtype=dtype)
        X_eval.requires_grad_(True)

        # 3. Pre-calculate variances for Normalization
        var_x = torch.var(train_x, dim=0).detach() + 1e-9 

        dgsm_estimates = torch.zeros((self.B, d), device=device) # Shape: (B, d) -> We assume B (bootstrap) is 1 per realization or synonymous with NZ loop

        # 4. Compute Derivatives over GP Realizations
        for k in range(self.B):
            
            output_dist = metamodel.likelihood(metamodel(X_eval))
            
            # Sample a single posterior function realization
            # shape: (M,) (M samples of the function)
            f_sample = output_dist.rsample(sample_shape=torch.Size([1])).squeeze(0)

            var_y_pred = torch.var(f_sample).detach() + 1e-9

            # Compute gradients: d(f_sample) / d(X_eval)
            grad_outputs = torch.ones_like(f_sample)
            grads = torch.autograd.grad(
                outputs=f_sample, 
                inputs=X_eval, 
                grad_outputs=grad_outputs,
                create_graph=False, # We don't need 2nd derivatives
                retain_graph=False,
                only_inputs=True
            )[0] # Shape: (M, d)
            

            # Calculate DGSM (unnormalized): E[(df/dx)^2]
            nu_i = torch.mean(grads**2, dim=0) # Average over M spatial samples -> (d,)

            # Normalize to match Sobol scale: nu_i * Var(X) / Var(Y)
            S_proxy = (nu_i * var_x) / var_y_pred
            
            dgsm_estimates[k, :] = S_proxy

            # Zero out gradients for next loop (safety, though X_eval is re-used)
            if X_eval.grad is not None:
                X_eval.grad.zero_()

        # 5. Statistics & Output
        dgsm_np = dgsm_estimates.cpu().numpy()

        dgsm_mean = np.mean(dgsm_np, axis=0)
        dgsm_var = np.var(dgsm_np, axis=0, ddof=1)

        # 95% Confidence Intervals
        confidence_level = 0.95
        alpha = 1 - confidence_level
        lower = np.percentile(dgsm_np, alpha / 2 * 100, axis=0)
        upper = np.percentile(dgsm_np, (1 - alpha / 2) * 100, axis=0)
        confidence_intervals = np.column_stack([lower, upper])

        print(f'Results for interactions of derivative-based sensitivity measure (DGSM) method')
        print(f"Effectively an upper bound on the total order sobol indices \n")
        print(f'predicted mean: {dgsm_mean} and var: {dgsm_var}')

    def interactions_tt(self, train_x, train_y, metamodel):
        """
        Calculate higher-order Sobol indices using Tensor Train (TT) decomposition.

        Method:
            1. Uses TT-Cross approximation to compress the GP Posterior Mean into a TT.
            2. Computes Sobol indices analytically from the TT cores.
            3. Approximates 'High Order' as S_total - S_1.

        Note:
            This method analyzes the deterministic 'Mean' surface of the GP. 
            Variance and CIs are returned as zeros because the TT-Cross is 
            performed once on the expected predictor.

        Args:
            train_x: Tensor (n, d)
            train_y: Tensor (n, 1)
            metamodel: GPyTorch model

        Returns:
            tt_mean: (d,) array of high-order interaction indices
            tt_var: (d,) zeros (deterministic approximation)
            confidence_intervals: (d, 2) zeros
        """

        # 1. Setup & Optimization
        d = self.problem['num_vars']
        device = train_x.device
        dtype = train_x.dtype

        # Train/Optimize the metamodel
        metamodel, metamodel.likelihood, _ = optimize(metamodel, train_x, train_y)
        metamodel.eval(); metamodel.likelihood.eval()

        # 2. Define Function Wrapper for TT-Cross
        # tntorch requires a function f(x) -> y where x is (batch, d)
        # We wrap the GP posterior mean.
        def gp_mean_func(*args):
            """
            Supports both:
            f(x)              where x is (batch, d)
            f(x1, x2, ..., xd) where each xi is (batch,)
            """
            with torch.no_grad():

                # ---- Case 1: f(x) ----
                if len(args) == 1:
                    x = args[0]

                # ---- Case 2: f(x1, x2, ..., xd) ----
                else:
                    # Stack per-dimension inputs into (batch, d)
                    x = torch.stack(args, dim=-1)

                # ---- Normalize x ----
                if isinstance(x, list):
                    x = torch.as_tensor(x, dtype=dtype)
                elif isinstance(x, np.ndarray):
                    x = torch.as_tensor(x, dtype=dtype)
                elif isinstance(x, torch.Tensor):
                    x = x.to(dtype=dtype)
                else:
                    raise TypeError(f"Unsupported input type: {type(x)}")

                # Ensure shape (batch, d)
                if x.ndim == 1:
                    x = x.unsqueeze(0)

                # ---- GP posterior mean ----
                output = metamodel.likelihood(metamodel(x))
                return output.mean.squeeze(-1)
            
        # Define hyper-rectangle domain from problem bounds
        bounds_np = np.array(self.problem['bounds'])
        domain = torch.as_tensor(bounds_np, dtype=dtype)

        # TT-Cross Approximation
        # ranks_max: controls the complexity of interactions captured
        # n: number of grid points per dimension (discretization)
        tt = tn.cross(
            function=gp_mean_func,
            domain=domain,
            rmax=10,       # Increase if interactions are very complex
            #n=32,               # 32-64 is usually sufficient for smooth GPs
            #tol=1e-3,           # Approximation tolerance
            #device=device,
            verbose=False
        )

        def sobol_from_tt(tt):
            """
            Computes first-order and total Sobol indices from a TT surrogate.
            """
            d = tt.dim()

            # Total variance
            mean = tt.integrate()
            var = tt.integrate(lambda x: (x - mean) ** 2)

            S1 = torch.zeros(d)
            ST = torch.zeros(d)

            for i in range(d):
                # First-order: Var(E[f | xi]) / Var(f)
                tt_i = tt.marginalize(dims=[j for j in range(d) if j != i])
                mean_i = tt_i.integrate()
                S1[i] = tt_i.integrate(lambda x: (x - mean_i) ** 2) / var

                # Total-order: 1 - Var(E[f | x_-i]) / Var(f)
                tt_not_i = tt.marginalize(dims=[i])
                mean_not_i = tt_not_i.integrate()
                ST[i] = 1 - tt_not_i.integrate(lambda x: (x - mean_not_i) ** 2) / var

            return S1, ST


        # 4. Compute Sobol Indices from TT
        #sobol_stats = tn.sensitivity.sobol_indices(tt)
        
        # Extract indices (tntorch returns 1D tensors)
        S1, ST = sobol_from_tt(tt)
        S1 = S1.cpu().numpy()
        ST = ST.cpu().numpy()
        
        # Calculate Higher Order Interactions proxy
        high_order_mean = ST - S1
        
        # 5. Returns
        # Variance and CI are 0 because we analyzed the deterministic mean surface
        high_order_var = np.zeros_like(high_order_mean)
        confidence_intervals = np.zeros((d, 2))

        print(f'Results for interactions of Sobol Tensor Train (TT) Decomposition method')
        print(f'predicted TT Higher-Order (S_T - S_1): {high_order_mean} and var: {high_order_var}')

    def interactions_asm(self, train_x, train_y, metamodel):
        """
        Perform Active Subspace Method (ASM) analysis to detect ridge structures and interactions.
        
        Logic:
            1. Compute gradient covariance matrix C = E[ (df/dx)^T (df/dx) ]
            2. Compute eigenvalues (activity) and eigenvectors (active directions).
            3. Activity Scores (diagonal of C) are returned as sensitivity proxy.
            4. Eigenvector structure is analyzed for axis-alignment (separability).

        Args:
            train_x, train_y: Tensor data
            metamodel: GPyTorch model

        Returns:
            activity_scores: (d,) numpy array (proxy for Total Sensitivity)
            activity_var: (d,) zeros (deterministic analysis)
            confidence_intervals: (d, 2) zeros
        """
        
        # 1. Setup & Optimization
        d = self.problem['num_vars']
        self.NZ = train_x.shape[0]
        device = train_x.device
        dtype = train_x.dtype

        # Train/Optimize
        metamodel, metamodel.likelihood, _ = optimize(metamodel, train_x, train_y)
        metamodel.eval(); metamodel.likelihood.eval()

        # 2. Monte Carlo Sampling for Gradient Estimation
        # We sample the space uniformly/Sobol to approximate the expectation E[...]
        X_eval_np = sobol_sampler.sample(self.problem, self.M, calc_second_order=False)
        X_eval = torch.tensor(X_eval_np, device=device, dtype=dtype)
        X_eval.requires_grad_(True)

        # 3. Compute Gradients over GP Realizations
        # Accumulate the C matrix: sum of outer products of gradients
        C = torch.zeros((d, d), device=device)
        
        # We average C over NZ realizations to get a robust estimate
        for k in range(self.B):
            output_dist = metamodel.likelihood(metamodel(X_eval))
            f_sample = output_dist.rsample(sample_shape=torch.Size([1])).squeeze(0)

            # Compute gradients: shape (M, d)
            # sum() trick is efficient for batch gradients of independent samples
            grads = torch.autograd.grad(
                outputs=f_sample.sum(), 
                inputs=X_eval, 
                create_graph=False
            )[0]
            
            # Update C: (1/M) * (Grads.T @ Grads)
            # We average over samples M
            C_k = torch.matmul(grads.T, grads) / self.M
            C += C_k

            # Zero grads
            if X_eval.grad is not None:
                X_eval.grad.zero_()

        # Average over GP Bootstrap realizations
        C = C / self.B

        # 4. Eigendecomposition
        # Eigenvalues (lambda) and Eigenvectors (W)
        # eigh returns them in ascending order
        eigvals, eigvecs = torch.linalg.eigh(C)
        
        # Sort descending
        eigvals = torch.flip(eigvals, dims=[0])
        eigvecs = torch.flip(eigvecs, dims=[1])

        # 5. Metrics Calculation
        
        # A) Activity Scores (Sensitivity Proxy)
        # This is mathematically equivalent to the diagonal of C (DGSM)
        # But derived from the subspace view: sum(lambda_j * w_ij^2)
        activity_scores = torch.diag(C).detach().cpu().numpy()

        # B) Interaction Detection (Ridge Structure Analysis)
        # We analyze the first dominant eigenvector (w_1)
        w1 = eigvecs[:, 0].detach().cpu().numpy()
        w1_sq = w1**2
        
        print("\n--- Active Subspace Analysis ---")
        #print(f"Top 3 Eigenvalues: {eigvals[:3].detach().cpu().numpy()}")
        print(f"Eigenvectors: {w1_sq}")
        
        # Heuristic: If energy is spread across multiple components, we have mixing.
        # Check if the dominant eigenvector is aligned with a single axis.
        max_component = np.max(w1_sq)
        dominant_idx = np.argmax(w1_sq)
        
        if max_component > 0.9: # Threshold for "Axis Aligned"
            print(f"Dominant direction is axis-aligned with x{dominant_idx} (Score: {max_component:.2f}).")
            print("Interpretation: Additive / Separable dominant structure.")
        else:
            print(f"Dominant direction is a linear combination (Max alignment: {max_component:.2f}).")
            print(f"Significant components: {np.where(w1_sq > 0.1)[0]}")
            print("Interpretation: Ridge structure detected (Interaction).")

        print(f'activity scores: {activity_scores}')

    def update_partition(self, interactions):
        """
        Partition dimensions using a greedy algorithm based on 2nd-order interactions.

        Inputs:
        - interactions: numpy array (1, d) matrix with high-order Sobol indices.
        Output:
        - partitions: list of lists, each sublist contains indices belonging to a partition.
        """
        d = self.problem['num_vars']

        # initialize stack of dimensions
        S = list(range(d))

        # P will hold the partitions
        P = []
        for x in range(d):
            if interactions[x] < self.epsilon:
                P.append([x])
            else:
                if len(P) == 0:
                    P.append([x])
                else:
                    P[0].append(x)
        return P
