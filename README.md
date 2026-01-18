# Additive Neurostimulation

## Overview

**Additive Neurostimulation** is a Bayesian optimization framework designed to address a dual challenge in neurostimulation research: **discovering the interaction structure of stimulation parameters** while simultaneously **improving optimization efficiency**.

Traditional Gaussian Process (GP) models used in Bayesian optimization treat all input dimensions as fully coupled, which can be sample-inefficient when the underlying function exhibits additive or partially separable structure. Conversely, purely additive GP kernels assume complete independence between dimensions, failing to capture genuine parameter interactions that are often present in neurostimulation responses.

This project introduces **SobolGP**, an adaptive algorithm that bridges this gap by leveraging **global sensitivity analysis** during the optimization process. Specifically, SobolGP:

1. **Learns the additive structure dynamically**: By computing Sobol sensitivity indices on a GP surrogate model, the algorithm identifies which input dimensions interact and which can be treated as independent. This information is used to construct a partitioned kernel that accurately reflects the true structure of the objective function.

2. **Improves sample efficiency**: Additive GP kernels propagate information more effectively across the input space, requiring fewer samples to build accurate surrogate models. By detecting and exploiting additive subspaces when they exist, SobolGP achieves faster convergence than standard GPs.

3. **Reveals parameter interaction patterns**: Beyond optimization, the predicted high-order Sobol indices provide interpretable insights into how neurostimulation parameters interact. This serves as a tool for **neuroscience discovery**, helping researchers understand which parameter combinations produce synergistic or antagonistic effects.

The proposed methods are benchmarked against **AdditiveGP** and **ExactGP** baselines on:
- **Synthetic black-box optimization functions**: Hartmann, Michalewicz, Ishigami, GSobol, and custom multi-modal test functions
- **Real neurostimulation datasets**: 2D non-human primate (NHP), 2D rat motor cortex, 2D spinal cord stimulation, and upcoming 5D rat experiments

## Literature

This project builds upon foundational work in Bayesian optimization, additive Gaussian processes, and global sensitivity analysis:

- **[Global sensitivity analysis based on Gaussian-process metamodelling for complex biomechanical problems](https://arxiv.org/abs/2202.01503)** (Wirthl et al., 2023) — Introduces the methodology for computing Sobol sensitivity indices using GP surrogate models, enabling variance-based sensitivity analysis for computationally expensive models.

- **[Additive Gaussian Processes Revisited](https://arxiv.org/abs/2206.09861)** (Lu et al., ICML 2022) — Proposes the orthogonal additive kernel (OAK) with identifiability constraints, connecting additive GPs to functional ANOVA decomposition.

- **[Discovering and Exploiting Additive Structure for Bayesian Optimization](https://proceedings.mlr.press/v54/gardner17a.html)** (Gardner et al., AISTATS 2017) — Presents an algorithm to automatically discover additive structure in objective functions and exploit it for more efficient Bayesian optimization.

- **[Autonomous optimization of neuroprosthetic stimulation parameters that drive the motor cortex and spinal cord outputs in rats and monkeys](https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(23)00118-0)** (Bonizzato et al., Cell Reports Medicine 2023) — Demonstrates GP-based Bayesian optimization for neurostimulation parameter tuning across multiple animal models and neural targets. This work provides the neurostimulation datasets used in this project.

## Installation

### Requirements
- Python 3.11+
- CUDA 11.8 (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/additive-neurostimulation.git
cd additive-neurostimulation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** If you encounter any installation issues, please contact the maintainers.

## Project Structure

```
additive-neurostimulation/
├── src/
│   ├── GPBO.py                      # Main Bayesian optimization runners (run_bo, run_partitionbo)
│   ├── neurostim.py                 # Neurostimulation experiment pipeline
│   ├── sobol_surrogate.py           # Surrogate-based Sobol computation
│   ├── models/
│   │   ├── gaussians.py             # GP models (ExactGP, AdditiveGP, SobolGP, MHGP)
│   │   └── sobols.py                # Sobol sensitivity analysis methods
│   ├── utils/
│   │   ├── synthetic_datasets.py    # Synthetic test functions (Hartmann, Michalewicz, etc.)
│   │   ├── sensitivity_utils.py     # Sensitivity analysis helpers
│   │   └── visualize_neurostim.py   # Visualization utilities
│   └── tests/
│       ├── test_model_computetimes.py
│       ├── parallelizedGP.py
│       └── params_analysis.ipynb
├── datasets/                        # Neurostimulation datasets (rat, nhp, spinal, 5d_rat)
└── output/                          # Experiment results and figures
```

## Quick Start

Run a benchmark experiment comparing ExactGP, AdditiveGP, and SobolGP on the Hartmann 6D function:

```bash
python src/GPBO.py --method optimization_metrics --f_ob hartmann --dim 6 --n_iter 200 --n_reps 20 --kappas 9.0,9.0,7.0
```

### Parameters

| Argument | Description |
|----------|-------------|
| `--method` | Experiment type: `run_bo`, `run_partitionbo`, `kappa_search`, `optimization_metrics`, `partition_reconstruction` |
| `--f_ob` | Test function: `hartmann`, `michalewicz`, `ackley`, `ishigami`, `gsobol`, `twoblobs`, `dblobs`, etc. |
| `--dim` | Input dimensionality |
| `--n_iter` | Optimization budget (number of iterations) |
| `--n_reps` | Number of repetitions for averaging |
| `--kappas` | UCB exploration weights for each model (ExactGP, AdditiveGP, SobolGP) |

Results are saved to `output/synthetic_experiments/<function_name>/`.
