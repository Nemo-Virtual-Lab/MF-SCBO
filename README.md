# MF-SCBO: Multi-Fidelity Scalable Constrained Bayesian Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains an implementation of **MF-SCBO**, an adaptation of the SCBO algorithm for **multi-fidelity constrained Bayesian optimization**.

The original SCBO algorithm ([Eriksson et al., 2021](https://proceedings.mlr.press/v130/eriksson21a/eriksson21a.pdf)) solves **high-dimensional constrained Bayesian optimization problems** using a **trust region approach**.  
Here, we extend SCBO using [BoTorch](https://botorch.org/) in Python to handle **multi-fidelity problems** efficiently. **Documentation** avaible [here](https://luplz.github.io/MF-SCBO/).

---

## 🚀 Installation

1. Clone this repository:

```bash
git clone https://github.com/luplz/MF-SCBO.git
cd MF-SCBO
```

2. Install the required dependencies:

 ```bash
pip install -r requirements.txt
```

3. Add the repository root to the PYTHONPATH

```bash
cd MF-SCBO
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## 🛠️ Usage Example


The repository contains example scripts in the [`examples/`](examples/) folder demonstrating how to use MF-SCBO with different functions and settings. 
**Note:** The optimisation problem must be formulated as a **maximisation** problem, and all constraints must be expressed as **inequalities of the form** $c(x) \leq 0$ For **single fidelity** optimization, define only list of single element.

1. Initialize and optimize

```python
from mfscbo.mfscbo_optimization import ScboMfOptimizer
import torch
import numpy as np
import matplotlib.pyplot as plt
from mfscbo.utils import eval_fun_unnormalized

# -----------------------------
# Torch arguments
# -----------------------------
torchargs = {"device": "cpu", "dtype": torch.double}

# -----------------------------
# Evaluation objectives (converted to maximization)
# -----------------------------
bounds = torch.tensor([[0.0]*2, [1.0]*2], **torchargs)
dim = bounds.shape[1]

f0_ = torch.sin(5 * np.pi * x[:, 0]
f1_ = lambda x : torch.sin(5 * np.pi * x[:, 0]) + (x[:, 1] - 0.5)**2
f2_ = lambda x : torch.sin(5 * np.pi * x[:, 0]) + (x[:, 1] - 0.5)**2 + 0.5 * torch.sin(10 * np.pi * x[:, 0])

#Rewrite as maximization pb because of scbo (and unnormalize)
f0 = lambda x : -eval_fun_unnormalized(x, f0_, bounds)[:,None] #shape (N, 1)
f1 = lambda x : -eval_fun_unnormalized(x, f1_, bounds)[:,None] #shape (N, 1)
f2 = lambda x : -eval_fun_unnormalized(x, f2_, bounds)[:,None] #shape (N, 1)
eval_objectives = [f0, f1, f2]


# -----------------------------
# Constraints
# -----------------------------
constraints = lambda x : (x[:, 0]**2 + x[:, 1]**2 - 1.0).unsqueeze(-1) #shape (N,nb_constraints), here nb_constraints=1
def log_constraints(x) : #bilog transformation to improve, see scbo article
        c = constraints(x)
        return torch.sign(c) * torch.log(1 + torch.abs(c))

def eval_constraints(x) : #unnormalization of constraints
    return eval_fun_unnormalized(x, log_constraints, bounds)

# -----------------------------
# Optimization parameters
# -----------------------------
n_pts = [12, 9, 3]
rhos = [1, 1, 1]
costs = [1, 10, 20]

# -----------------------------
# Initialize and run optimizer
# -----------------------------
optimizer = ScboMfOptimizer(
    eval_objectives=eval_objectives,
    constraints=eval_constraints,
    rhos=rhos,
    costs=costs,
    dim=dim,
    n_pts=n_pts,
    bounds=bounds,
    type_of_batch="parallel",
    type_of_centering="evaluated",
    torchargs=torchargs,
    restart=True,
    test=True #if true, in predictive mode, the best_Y is then computed with high-fidelity evaluations
)

optimizer.initialize()
x, y = optimizer.optimize(max_cost=2000, batch_size=16, path="results/test", verbose=True)
```

2. Load and post-process results
   
```python
# -----------------------------
# Load results
# -----------------------------
data = np.load("results/test/MFSCBO_data.npz", allow_pickle=True)
gps_info = data["gps_info"]
iterations = data["iterations"]
xvalues = data["cost_iter"]
bestY = -data["best_Y"] #for minimization plot
best_C = data["best_C"]
nbiters = data["nb_iter"]
tr_lengths = data["trust_region_lengths"]

# -----------------------------
# Post-process results
# -----------------------------
best_violated_C = np.sum(np.maximum(0, best_C), axis=1) #bestY when constraints are not respected 
bestY = np.array(bestY, dtype=np.float64)
running_max = np.max(bestY)
bestY = np.where(best_violated_C > 0, running_max, bestY)

rhos_values = [[1] for _ in range(len(gps_info)-1)]
for fid in range(1, len(gps_info)):
    for it in range(len(iterations)-1):
        rhos_values[fid-1].append(gps_info[fid][it]["rho"])

noises = [[] for _ in range(len(gps_info))]
for fid in range(len(gps_info)):
    for it in range(len(iterations)-1):
        noises[fid].append(gps_info[fid][it]["noise"])
```

3. Plot results
   
```python
from mfscbo.plot import plot_rhos, plot_noises, plot_bestYs, plot_nbiters
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# -----------------------------
# Plots
# -----------------------------
fig = plt.figure(figsize=(15,12))  
gs = GridSpec(3, 2, figure=fig)  
ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[2,0])
ax5 = fig.add_subplot(gs[2,1])

plot_bestYs(bestY, xvalues, ax=ax1)
plot_rhos(rhos, xvalues, ax=ax2)
plot_noises(noises, xvalues[1:], ax=ax3) #there is no gp at initial iteration 
plot_nbiters(nbiters, xvalues, ax=ax4)
plot_tr_lengths(tr_lengths, xvalues, ax=ax5)

ax1.set_title("Objective value")
ax1.grid()
ax2.set_title(r"$\rho$ values")
ax2.legend()
ax2.grid()
ax3.set_title("Noise values")
ax3.legend()
ax3.grid()
ax3.set_yscale("log")
ax4.set_title("Number of fidelity evaluations")
ax4.legend()
ax4.grid()
ax5.set_title("Trust-region lengths")
ax5.legend()
ax5.grid()

plt.tight_layout()
plt.savefig("results/test/MFSCBO_plots.pdf")
plt.show()

```

## 📂 Data

### `MFSCBO_data.npz`

Contains the main results of the optimization:

- `gps_info` (`List[dict]`): GP model information per fidelity level
- `generated_X`: List of generated points per iteration 
- `generated_Y`: List of generated values per iteration 
- `generated_C`: List of generated constraint values per iteration 
- `criterions`: List of criterion values for each iteration
- `trust_region_lengths`: List of trust region lengths per iteration 
- `best_X`: Best points according to `best_Y`
- `best_Y`: Best values per iteration 
- `best_C`: Constraint values corresponding to `best_Y` 
- `iterations`: Iteration numbers 
- `nb_iter`: Number of iterations for each fidelity level 
- `cost_iter`: Total costs for each iteration 

### `MFSCBO_info.npz`

Contains configuration and meta-information about the optimization run:

- `type_of_centering` (`str`): Type of centering ("evaluated" or "predicted")
- `type_of_batch` (`str`): Type of batch ("parallel" or "sequential")
- `batch_size` (`int`): Batch size used
- `max_cost` (`float`): Maximum total cost allowed
- `restart` (`bool`): Whether the optimizer restarts when the trust region becomes too small
- `test` (`bool`): Whether the optimizer is in test mode (evaluates best points with high-fidelity objectives)

### `MFSCBO_log.txt`

Contains all verbose information printed during the optimization run:

- Tracks the progress of the optimizer, including iteration info, GP updates, rho values, noise levels, and evaluated points.
