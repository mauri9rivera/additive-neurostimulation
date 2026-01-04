import torch
import torch.multiprocessing as mp
import math
import time
import os
from concurrent.futures import ThreadPoolExecutor
import queue
import threading


# ==========================================
# Part 1: Kernel Definitions (Ground Up)
# ==========================================

class RBFKernel:
    """
    Standard RBF Kernel: k(x, x') = sigma^2 * exp(- ||x - x'||^2 / (2 * l^2))
    """
    def __init__(self, lengthscale=1.0, variance=1.0, device='cpu'):
        self.raw_lengthscale = torch.tensor([math.log(lengthscale)], device=device, requires_grad=True)
        self.raw_variance = torch.tensor([math.log(variance)], device=device, requires_grad=True)
        self.device = device

    @property
    def lengthscale(self):
        return torch.exp(self.raw_lengthscale)

    @property
    def variance(self):
        return torch.exp(self.raw_variance)

    def forward(self, x1, x2):
        # Optimized distance for 1D inputs (N, 1)
        # (x1 - x2)^2 = x1^2 + x2^2 - 2x1x2
        x1_sq = x1.pow(2)
        x2_sq = x2.pow(2)
        dist_sq = x1_sq.view(-1, 1) + x2_sq.view(1, -1) - 2.0 * torch.mm(x1, x2.t())
        dist_sq = dist_sq.clamp(min=0.0)
        K = self.variance * torch.exp(-0.5 * dist_sq / self.lengthscale.pow(2))
        return K

    def parameters(self):
        return [self.raw_lengthscale, self.raw_variance]

# ==========================================
# Part 2: Exact GP (Baseline)
# ==========================================

class ExactGP:
    def __init__(self, train_x, train_y, noise_variance=1e-4, device='cpu'):
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.raw_lengthscale = torch.tensor([1.0], device=device, requires_grad=True) 
        self.raw_output_scale = torch.tensor([1.0], device=device, requires_grad=True)
        self.raw_noise = torch.tensor([math.log(noise_variance)], device=device, requires_grad=True)
        self.device = device

    @property
    def lengthscale(self): return torch.exp(self.raw_lengthscale)
    @property
    def output_scale(self): return torch.exp(self.raw_output_scale)
    @property
    def noise(self): return torch.exp(self.raw_noise)

    def forward_kernel(self, x):
        # Full D-dimensional distance
        dist_sq = torch.cdist(x, x, p=2).pow(2)
        K = self.output_scale * torch.exp(-0.5 * dist_sq / self.lengthscale.pow(2))
        return K

    def negative_log_likelihood(self):
        K = self.forward_kernel(self.train_x)
        jitter = 1e-4
        # Add the learnable noise
        K_noise = K + torch.eye(K.size(0), device=self.device) * self.noise

        K_noise = K_noise + torch.eye(K.size(0), device=self.device) * jitter
        # ---------------------------------

        try:
            L = torch.linalg.cholesky(K_noise)
        except torch.linalg.LinAlgError:
            # Fallback: If it still fails, add more jitter and try once more
            # This often happens during early optimization steps
            K_noise = K_noise + torch.eye(K.size(0), device=self.device) * 1e-3
            L = torch.linalg.cholesky(K_noise)

        alpha = torch.cholesky_solve(self.train_y.unsqueeze(1), L)
        
        # LML terms
        data_fit = -0.5 * torch.matmul(self.train_y.unsqueeze(0), alpha)
        complexity = -torch.sum(torch.log(torch.diagonal(L)))
        constant = -0.5 * self.train_x.size(0) * math.log(2 * math.pi)
        return -(data_fit + complexity + constant)

    def parameters(self):
        return [self.raw_lengthscale, self.raw_output_scale, self.raw_noise]

# ==========================================
# Part 3: Additive GP (Threading)
# ==========================================

class AdditiveGP:
    """
    Additive GP: K = sum(K_d).
    Implements Serial, Threaded, and Multiprocess strategies.
    """
    def __init__(self, train_x, train_y, mode='serial', devices=None):
        self.mode = mode
        self.N, self.D = train_x.shape
        self.devices = devices if devices else ['cuda:0' if torch.cuda.is_available() else 'cpu']
        self.primary_device = 'cpu'
        self.train_y = train_y.to(self.primary_device)
        self.train_x = train_x
        self.train_x_map = {}
        
        # Initiate sub-kernels
        self.sub_kernels = []
        
        if mode == 'multiprocess_gpu':
            self.dim_map = {d: self.devices[d % len(self.devices)] for d in range(self.D)}
            for dev in self.devices:
                self.train_x_map[dev] = train_x.to(dev)
            for d in range(self.D):
                self.sub_kernels.append(RBFKernel(device=self.dim_map[d]))
        else:
            self.train_x_map[self.primary_device] = train_x.to(self.primary_device)
            for d in range(self.D):
                self.sub_kernels.append(RBFKernel(device=self.primary_device))

        self.raw_noise = torch.tensor([math.log(1e-4)], device=self.primary_device, requires_grad=True)

    @property
    def noise(self): return torch.exp(self.raw_noise)


    def compute_loss_and_gradients(self):
        """
        Orchestrates the Parallel Forward and Parallel Backward passes.
        """
        if self.mode == 'multiprocess_gpu':
            return self._manual_parallel_step()
        else:
            # Fallback to standard PyTorch autograd for simple/serial modes
            return self._negative_log_likelihood()


    def compute_kernel(self):
        if self.mode == 'serial':
            K = torch.zeros((self.N, self.N), device=self.primary_device)
            for d in range(self.D):
                xd = self.train_x[:, d].view(-1, 1).to(self.primary_device)

                K += self.sub_kernels[d].forward(xd, xd)
            return K
            
        elif self.mode == 'threaded':
            # Threaded execution on single device (relying on CUDA streams or CPU cores)
            def calc_d(d):
                xd = self.train_x[:, d].view(-1, 1).to(self.primary_device)
                return self.sub_kernels[d].forward(xd, xd)
            
            with ThreadPoolExecutor(max_workers=24) as exe:
                # Dispatch all
                futures = list(exe.map(calc_d, range(self.D)))
            
            K = sum(futures) # Summing tensors
            return K


    def _negative_log_likelihood(self):
        K = self.compute_kernel()

        jitter = 1e-4
        # Add the learnable noise
        K_noise = K + torch.eye(K.size(0), device=self.primary_device) * self.noise

        K_noise = K_noise + torch.eye(K.size(0), device=self.primary_device) * jitter
        # ---------------------------------

        try:
            L = torch.linalg.cholesky(K_noise)
        except torch.linalg.LinAlgError:
            # Fallback: If it still fails, add more jitter and try once more
            # This often happens during early optimization steps
            K_noise = K_noise + torch.eye(K.size(0), device=self.primary_device) * 1e-3
            L = torch.linalg.cholesky(K_noise)

 
        
        alpha = torch.cholesky_solve(self.train_y.unsqueeze(1), L)
        data_fit = -0.5 * torch.matmul(self.train_y.unsqueeze(0), alpha)
        complexity = -torch.sum(torch.log(torch.diagonal(L)))
        constant = -0.5 * self.N * math.log(2 * math.pi)
        
        return -(data_fit + complexity + constant)


    def _manual_parallel_step(self):
        """
        Implements the Gradient Decomposition Method.
        1. Workers compute K_local -> send DETACHED tensor to Primary.
        2. Primary sums K -> computes Loss -> computes dLoss/dK.
        3. Primary sends dLoss/dK back to workers.
        4. Workers re-run forward (to build local graph) and backward with incoming grad.
        """
        
        # --- Phase 1: Distributed Forward (Detach to break graph) ---
        def forward_worker(dev):
            dims = [d for d in range(self.D) if self.dim_map[d] == dev]
            local_K = torch.zeros((self.N, self.N), device=dev)
            for d in dims:
                xd = self.train_x_map[dev][:, d].view(-1, 1)
                # We compute inside torch.no_grad() or simply detach immediately
                # because we don't want the graph on the primary device.
                with torch.no_grad():
                    local_K += self.sub_kernels[d].forward(xd, xd)
            return local_K

        with ThreadPoolExecutor(max_workers=len(self.devices)) as exe:
            fwd_futures = {dev: exe.submit(forward_worker, dev) for dev in self.devices}
            # Gather detached tensors
            results = {dev: f.result() for dev, f in fwd_futures.items()}

        # --- Phase 2: Global Loss & Matrix Gradient ---
        # Sum detached kernels
        K_global = torch.zeros((self.N, self.N), device=self.primary_device)
        for dev, k_local in results.items():
            K_global += k_local.to(self.primary_device)
        
        # IMPORTANT: Enable grad tracking on the aggregated matrix
        K_global.requires_grad_(True)
        
        # Compute NLL (Standard Exact GP logic)
        loss = self._calc_nll(K_global)
        
        # Compute dLoss / dK_global
        # This gives us the gradient to scatter back to workers
        grad_K_global = torch.autograd.grad(loss, K_global)[0]

        # Also populate noise gradient (local to primary)
        # We need to manually backward the noise parameter or handle it separately
        # (Already handled by autograd.grad if we included raw_noise in calculation, 
        # but requires_grad=True on K_global blocks propagation to raw_noise unless we are careful.
        # Simplest: Compute dL/dNoise manually or separate the graph).
        # For simplicity here: we calculate noise grad via a small graph on primary:
        loss_for_noise = self._calc_nll(K_global.detach()) 
        loss_for_noise.backward() # This populates self.raw_noise.grad
        
        # --- Phase 3: Distributed Backward ---
        def backward_worker(dev, global_grad_chunk):
            # 1. Move global gradient to local device
            local_grad = global_grad_chunk.to(dev)
            
            # 2. Re-compute local forward to build the computation graph 
            # (We must do this because we didn't store the graph in Phase 1 to save memory)
            dims = [d for d in range(self.D) if self.dim_map[d] == dev]
            local_K_with_graph = torch.zeros((self.N, self.N), device=dev)
            
            for d in dims:
                xd = self.train_x_map[dev][:, d].view(-1, 1)
                local_K_with_graph += self.sub_kernels[d].forward(xd, xd)
            
            # 3. Backward with incoming gradient
            local_K_with_graph.backward(local_grad)
            return True

        with ThreadPoolExecutor(max_workers=len(self.devices)) as exe:
            # We broadcast the SAME global gradient matrix to all workers
            # (Because dL/dK_sum = dL/dK_sub)
            bwd_futures = [
                exe.submit(backward_worker, dev, grad_K_global) 
                for dev in self.devices
            ]
            # Ensure all finished
            [f.result() for f in bwd_futures]

        return loss.item()

    def _calc_nll(self, K):
        # Adds noise and computes Cholesky
        K_noise = K + torch.eye(self.N, device=self.primary_device) * self.noise
        jitter = 1e-4
        K_noise = K_noise + torch.eye(K.size(0), device=self.primary_device) * jitter
        try:
            L = torch.linalg.cholesky(K_noise)
        except torch.linalg.LinAlgError:
            # Fallback: If it still fails, add more jitter and try once more
            # This often happens during early optimization steps
            K_noise = K_noise + torch.eye(K.size(0), device=self.primary_device) * 1e-2
            L = torch.linalg.cholesky(K_noise)
        
        alpha = torch.cholesky_solve(self.train_y.unsqueeze(1), L)
        data_fit = -0.5 * torch.matmul(self.train_y.unsqueeze(0), alpha)
        complexity = -torch.sum(torch.log(torch.diagonal(L)))
        constant = -0.5 * self.N * math.log(2 * math.pi)
        return -(data_fit + complexity + constant)

    def get_parameters(self):
        p = [self.raw_noise]
        for k in self.sub_kernels:
            p.extend(k.parameters())
        return p

# ==========================================
# Part 4: AdditiveGP (Processing)
# ==========================================

class GPUWorker(threading.Thread):
    def __init__(self, device, sub_kernels, train_x_local, out_queue):
        super().__init__()
        self.device = device
        self.sub_kernels = sub_kernels # List of RBFKernel objects on this device
        self.train_x_local = train_x_local # Pre-sliced tensor on this device
        
        self.in_queue = queue.Queue()
        self.out_queue = out_queue
        self.daemon = True # Kills thread if main process dies
        self.running = True

    def run(self):
        while self.running:
            # Block until we get a command
            try:
                task = self.in_queue.get() 
            except queue.Empty:
                continue

            cmd, data = task

            if cmd == 'TERMINATE':
                self.running = False
                self.in_queue.task_done()
                break

            elif cmd == 'FORWARD':
                # --- Phase 1: Compute Local Kernel ---
                # We do this inside no_grad because we only need the numerical value
                # on the primary device right now.
                with torch.no_grad():
                    local_K = torch.zeros((self.train_x_local.shape[0], self.train_x_local.shape[0]), device=self.device)
                    # Iterate over the dimensions assigned to this worker
                    # Note: train_x_local has shape (N, D_local)
                    for i, kernel in enumerate(self.sub_kernels):
                        xd = self.train_x_local[:, i].view(-1, 1)
                        local_K += kernel.forward(xd, xd)
                
                # Send detached result to main thread
                self.out_queue.put((self.device, local_K))
                self.in_queue.task_done()

            elif cmd == 'BACKWARD':
                # --- Phase 2: Recompute & Backward ---
                global_grad = data
                
                # 1. Move global gradient to this device
                local_grad = global_grad.to(self.device)
                
                # 2. Re-build graph (Local Forward)
                # We must re-run the forward pass to establish the graph connections
                # for autograd.
                local_K_graph = torch.zeros((self.train_x_local.shape[0], self.train_x_local.shape[0]), device=self.device)
                for i, kernel in enumerate(self.sub_kernels):
                    xd = self.train_x_local[:, i].view(-1, 1)
                    local_K_graph += kernel.forward(xd, xd)
                
                # 3. Local Backward
                # This populates .grad on self.sub_kernels parameters
                local_K_graph.backward(local_grad)
                
                # Signal completion
                self.out_queue.put((self.device, 'DONE'))
                self.in_queue.task_done()

class AdditiveGPQueue:
    def __init__(self, train_x, train_y, devices=None):
        self.N, self.D = train_x.shape
        self.devices = devices if devices else ['cuda:0' if torch.cuda.is_available() else 'cpu']
        self.primary_device = 'cpu' #self.devices[0]
        self.train_y = train_y.to(self.primary_device)
        
        # Jitter for stability
        self.jitter = 1e-3
        
        # 1. Initialize Sub-Kernels & Workers
        self.workers = []
        self.sub_kernels_flat = [] # Keep ref for optimizer
        self.result_queue = queue.Queue()
        
        # Distribute dimensions
        # Simple round-robin assignment
        self.dim_map = {d: self.devices[d % len(self.devices)] for d in range(self.D)}
        
        for dev_idx, dev in enumerate(self.devices):
            # Identify which dimensions belong to this device
            dims_on_dev = [d for d in range(self.D) if self.dim_map[d] == dev]
            
            if not dims_on_dev: continue
            
            # Create Kernels for this device
            dev_kernels = []
            for _ in dims_on_dev:
                k = RBFKernel(device=dev)
                dev_kernels.append(k)
                self.sub_kernels_flat.append(k)
            
            # Slice data for this device
            dev_x = train_x[:, dims_on_dev].to(dev)
            
            # Launch Worker
            worker = GPUWorker(dev, dev_kernels, dev_x, self.result_queue)
            worker.start()
            self.workers.append(worker)

        # Global Noise Parameter (on primary)
        self.raw_noise = torch.tensor([math.log(0.1)], device=self.primary_device, requires_grad=True)

    @property
    def noise(self): return torch.exp(self.raw_noise)

    def compute_loss_and_gradients(self):
        # --- Step 1: Trigger Forward on all Workers ---
        for w in self.workers:
            w.in_queue.put(('FORWARD', None))
            
        # --- Step 2: Gather Kernel Parts ---
        K_global = torch.zeros((self.N, self.N), device=self.primary_device)
        results_received = 0
        
        while results_received < len(self.workers):
            dev, k_part = self.result_queue.get()
            K_global += k_part.to(self.primary_device)
            results_received += 1
            
        # --- Step 3: Global Loss & Gradient on Primary ---
        K_global.requires_grad_(True)
        
        # Manual NLL calculation
        K_noise = K_global + torch.eye(self.N, device=self.primary_device) * (self.noise)
        K_noise += torch.eye(K_global.size(0), device=self.primary_device) * self.jitter
        
        try:
            L = torch.linalg.cholesky(K_noise)
        except torch.linalg.LinAlgError:
            # Fallback stability
            K_noise += torch.eye(self.N, device=self.primary_device) * 1e-2
            L = torch.linalg.cholesky(K_noise)
            
        alpha = torch.cholesky_solve(self.train_y.unsqueeze(1), L)
        data_fit = -0.5 * torch.matmul(self.train_y.unsqueeze(0), alpha)
        complexity = -torch.sum(torch.log(torch.diagonal(L)))
        constant = -0.5 * self.N * math.log(2 * math.pi)
        loss = -(data_fit + complexity + constant)
        
        # Calculate Gradient w.r.t Global Kernel
        grad_K_global = torch.autograd.grad(loss, K_global)[0]
        
        # Handle Noise Gradient (Since K_global detached noise from graph)
        # We compute noise grad locally on primary
        loss_noise_only = self._calc_nll_detached(K_global.detach())
        loss_noise_only.backward()

        # --- Step 4: Trigger Backward on all Workers ---
        # We assume the gradient is the same for all (linear sum property)
        for w in self.workers:
            w.in_queue.put(('BACKWARD', grad_K_global))
            
        # --- Step 5: Wait for completion ---
        finished_count = 0
        while finished_count < len(self.workers):
            self.result_queue.get() # Waiting for 'DONE' signals
            finished_count += 1
            
        return loss.item()

    def _calc_nll_detached(self, K):
        # Helper for noise gradient calculation
        K_noise = K + torch.eye(self.N, device=self.primary_device) * (self.noise + self.jitter)
        L = torch.linalg.cholesky(K_noise)
        alpha = torch.cholesky_solve(self.train_y.unsqueeze(1), L)
        data_fit = -0.5 * torch.matmul(self.train_y.unsqueeze(0), alpha)
        complexity = -torch.sum(torch.log(torch.diagonal(L)))
        return -(data_fit + complexity)

    def get_parameters(self):
        p = [self.raw_noise]
        for k in self.sub_kernels_flat:
            p.extend(k.parameters())
        return p
        
    def close(self):
        for w in self.workers:
            w.in_queue.put(('TERMINATE', None))
        for w in self.workers:
            w.join()


# ==========================================
# Part 5: Benchmarks
# ==========================================

def michalewicz_func(x, m=10):
    # Additive: sum( sin(xi) * sin(i xi^2 / pi)^2m )
    res = 0
    for i in range(x[0].shape[0]):
        xi = x[:, i]
        term = torch.sin(xi) * torch.pow(torch.sin((i + 1) * xi.pow(2) / math.pi), 2 * m)
        res -= term
    return res

def ackley_func(x):
    # Non-additive terms involved
    a, b, c = 20, 0.2, 2 * math.pi
    d = x.shape
    sum_sq = torch.sum(x.pow(2), dim=1)
    sum_cos = torch.sum(torch.cos(c * x), dim=1)
    return -a * torch.exp(-b * torch.sqrt(sum_sq/d)) - torch.exp(sum_cos/d) + a + math.e

# ==========================================
# Part 6: Runner
# ==========================================

def benchmark_dimensions():
    # Setup
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
    
    dims = [2, 5, 10, 50, 100,] 
    N = 1000
    modes = ['simple', 'serial', 'threaded']
    if torch.cuda.device_count() > 1:
        modes.append('multiprocess_gpu')
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        devices = None

    print(f"{'Dim':<5} | {'Mode':<15} | {'Fwd/Bwd Time (s)':<15}")
    print("-" * 40)

    for D in dims:
        # Data Gen
        X = torch.rand(N, D) * math.pi
        y = michalewicz_func(X)
        y = (y - y.mean()) / y.std() # Normalize

        for mode in modes:
            if mode == 'simple':
                gp = ExactGP(X, y, device='cpu')
                params = gp.parameters()
            elif mode == 'multiprocess_gpu':
                gp = AdditiveGPQueue(X, y, devices=devices)
                params = gp.get_parameters()
            else:
                gp = AdditiveGP(X, y, mode=mode, devices=devices)
                params = gp.get_parameters()
            
            optimizer = torch.optim.Adam(params, lr=0.1)
            
            # Timing Loop
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.time()

            for _ in range(50): # Short benchmark
                optimizer.zero_grad()
                
                if mode == 'simple':
                    loss = gp.negative_log_likelihood()
                    loss.backward()
                else:
                    # The magic happens here: Explicit Parallel Backward
                    gp.compute_loss_and_gradients()
                    
                optimizer.step()

            
            #if torch.cuda.is_available(): torch.cuda.synchronize()
            avg_time = (time.time() - start) / 5
            
            print(f"{D:<5} | {mode:<20} | {avg_time:.4f}")

def benchmark_queries():
    # Setup
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
    
    D = 2 
    n_iters = [100, 1000, 5000, 10000]
    modes = ['simple', 'serial', 'threaded']
    if torch.cuda.device_count() > 1:
        modes.append('multiprocess_gpu')
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        devices = None

    print(f"{'Queries':<5} | {'Mode':<15} | {'Fwd/Bwd Time (s)':<15}")
    print("-" * 40)

    for N in n_iters:
        # Data Gen
        X = torch.rand(N, D) * math.pi
        y = michalewicz_func(X)
        y = (y - y.mean()) / y.std() # Normalize

        for mode in modes:
            if mode == 'simple':
                gp = ExactGP(X, y, device='cpu')
                params = gp.parameters()
            elif mode == 'multiprocess_gpu':
                gp = AdditiveGPQueue(X, y, devices=devices)
                params = gp.get_parameters()
            else:
                gp = AdditiveGP(X, y, mode=mode, devices=devices)
                params = gp.get_parameters()
            
            optimizer = torch.optim.Adam(params, lr=0.1)
            
            # Timing Loop
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.time()

            for _ in range(50): # Short benchmark
                optimizer.zero_grad()
                
                if mode == 'simple':
                    loss = gp.negative_log_likelihood()
                    loss.backward()
                else:
                    # The magic happens here: Explicit Parallel Backward
                    gp.compute_loss_and_gradients()
                    
                optimizer.step()

            
            #if torch.cuda.is_available(): torch.cuda.synchronize()
            avg_time = (time.time() - start) / 5
            
            print(f"{N:<5} | {mode:<20} | {avg_time:.4f}")

if __name__ == "__main__":
    #benchmark_dimensions()
    benchmark_queries()