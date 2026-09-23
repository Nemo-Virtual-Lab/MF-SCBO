import os
import sys
import gc
import numpy as np
import copy 

import torch
from torch.quasirandom import SobolEngine
import gpytorch
from botorch.models import ModelListGP
from botorch.utils.transforms import unnormalize

from mfscbo.utils import Print_log_and_console, get_gp_info
from mfscbo.mfgp import DeltaGPModel, mean_var_of_fidelity_i, samples_of_fidelity_i, get_fitted_model, fit_model_delta
from mfscbo.scbo_state import ScboState, update_state, get_best_index_for_batch



## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Original code from BoTorch : https://botorch.org/docs/tutorials/scalable_constrained_bo/
# Some modifications have been made to adapt the code to  our needs.
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
Multi-fidelity Bayesian Optimization 
===========================================================================

Denote by 0 the low fidelity and S the high fidelity (so S+1 fidelities in total).
Model: f_1(x) = rho_1 * f_0(x) + delta_1(x),
       f_i(x) = rho_i * f_{i-1}(x) + delta_i(x)
       f_S(x) = rho_S * f_{S-1}(x) + delta_S(x)
 where
- f_i ~ GP(m_i, k_i)
- delta_i ~ GP(m_d_i, k_d_i)
Assume independence between GPs.

rho = [rho_0, rho_1, ..., rho_S] with rho_0 = 1
delta = [delta_0, delta_1, delta_2, ..., delta_S] with delta_0 = None
"""


def out_and_in(X_i, X_im1, rtol=1e-9, atol=1e-8):
    """Separate points X_i into points that are also in X_im1 and points that are not.
       The comparison is done by using : ||x - y|| <= atol + rtol * ||y|| => ||x-y||/||y|| <= rtol + atol/||y||

    Args:
        X_i (torch.Tensor): Points to separate of shape (n_i, dim)
        X_im1 (torch.Tensor): Reference points of shape (n_im1, dim)
        rtol (float, optional): Relative tolerance. Defaults to 1e-9.
        atol (float, optional): Absolute tolerance. Defaults to 1e-8.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (indices out according to X_im1 for i, indices in according to X_im1 for i, indices in according to X_im1 for im1)
    """
    diff = X_i.unsqueeze(1) - X_im1.unsqueeze(0)  # shape : (n_i, n_im1, dim) = (n_i, 1, dim) - (1, n_im1, dim)
    dist = diff.norm(dim=-1)                      #shape : (n_i, n_im1)

    ref_norm = X_im1.norm(dim=-1).unsqueeze(0)    #shape : (1, n_im1)
    tol = atol + rtol * ref_norm

    # Don't do in_mask_fid_i = in_mask.any(dim=1)  because multiple points can be close
    min_idx = dist.argmin(dim=0)  #shape : (n_im1)
    mask_of_dist = torch.zeros_like(dist, dtype=torch.bool)  #shape : (n_i, n_im1)
    mask_of_dist[min_idx, torch.arange(dist.shape[1])] = True
    dist_unique = torch.full_like(dist, float("inf"))  #shape : (n_i, n_im1)
    dist_unique[mask_of_dist] = dist[mask_of_dist]
    in_mask_fid_i = (dist_unique <= tol).any(dim=1)  #shape : (n_i)

    # Don't do in_mask_fid_im1 = in_mask.any(dim=0)  because multiple points can be close
    min_idx = dist.argmin(dim=1)  #shape : (n_i)
    mask_of_dist = torch.zeros_like(dist, dtype=torch.bool)  #shape : (n_i, n_im1)
    mask_of_dist[torch.arange(dist.shape[0]), min_idx] = True
    dist_unique = torch.full_like(dist, float("inf"))  #shape : (n_i, n_im1)
    dist_unique[mask_of_dist] = dist[mask_of_dist]
    in_mask_fid_im1 = (dist_unique <= tol).any(dim=0)  #shape : (n_im1)

    return ~in_mask_fid_i, in_mask_fid_i, in_mask_fid_im1

def get_initial_points_mf(dim, n_pts, torchargs, seed=None):
    """ Generate initial points for multi-fidelity optimization.

    Args:
        dim (int): Dimension of the problem.
        n_pts (List[int]): Number of initial points for each fidelity level :n_pts[i] is the number of i-th fidelity points (increasing order).
        torchargs (dict): Dictionary containing the device on which the tensors will be stored
                            and the dtype (device : str, dtype : torch.dtype)
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        List[torch.Tensor]: List of initial points for each fidelity level. The i-th element of the list
                            contains the initial points for the i-th fidelity of shape (n_pts[i], dim+1).
        
    """
    nb_fidelities = len(n_pts)
    fidelities = np.arange(nb_fidelities)

    #X_init in nb_fidelities parts
    X_init = []
    for n in n_pts:
        sobol = SobolEngine(dim, scramble=True, seed=seed)
        X_init.append(sobol.draw(n).to(**torchargs))

    #add a last column
    for fid in fidelities :
        X_init[fid] = torch.cat((X_init[fid], fid * torch.ones(n_pts[fid], 1, **torchargs)), dim=1)

    return X_init

def fidelity_selection(Y_next, means_next, time_costs, fidelities):
    """Select the best fidelity level based on the expected improvement.

    Args:
        Y_next (List[torch.Tensor]): The predicted values for the next points of shape (batch, nb_fidelities).
        means_next (List[torch.Tensor]): The predicted means for the next points of shape (batch, nb_fidelities).
        time_costs (List[float]): The time costs for each fidelity level of shape (nb_fidelities,).
        fidelities (List[int]): The list of fidelity levels to consider.

    Returns:
        (torch.Tensor, torch.Tensor): (The selected fidelity level for each point of shape (batch, 1), the corresponding terms for each fidelity level of shape (batch, nb_fidelities)).
    """
    terms = []
    for i in fidelities :
        term = torch.sqrt((Y_next[i] - means_next[i])**2) / time_costs[i]
        terms.append(term)
    terms = torch.stack(terms, dim=-1) #[batch, nb_fidelities]
    best_fidelity = torch.argmax(terms, dim=-1, keepdim=True) #[batch, 1]  
    return best_fidelity, terms

def generate_batch_mf(
    type_of_batch,
    state,
    model_lf,  
    models_delta,  
    rhos,  
    time_costs,   
    X_hf,  
    Y_hf,  
    C_hf,  
    batch_size,
    n_candidates,  
    constraint_model,
    sobol: SobolEngine,
    torchargs,
    ):

    """ Generate a batch of points to evaluate by sampling from the multi-fidelity model.

    Args:
        type_of_batch (str): Type of batch to generate (shoot of batch_size x n_candidates points then select the best ones : "parallel" or  the batch_size best points from n_candidates points : "sequential")
        state (ScboState): Current state of the optimization
        model_lf (botorch.models.SingleTaskGP): Fitted low fidelity GP model
        models_delta (List[botorch.models.SingleTaskGP]): Fitted delta GP models
        rhos (List[float]): Correlation parameter between fidelities
        time_costs (List[float]): Time costs of the fidelity functions
        X_hf (torch.Tensor): High fidelity points of shape (n_hf_pts, dim)
        Y_hf (torch.Tensor): High fidelity values of shape (n_hf_pts, 1)
        C_hf (torch.Tensor): Constraint values of shape (n_pts, n_constraints)
        batch_size (int): Number of points to sample
        n_candidates (int): Number of candidate points for thomson sampling
        constraint_model (botorch.models.ModelListGP): Fitted constraints GP models
        sobol (SobolEngine): Sobol engine for quasi-random sampling
        torchargs (dict): Dictionary containing the device on which the tensors will be stored
                            and the dtype (device : str, dtype : torch.dtype)

    Returns:  
        (torch.Tensor, torch.Tensor): (Sampled points of shape (batch_size, dim), criterion for fidelity selection of shape (batch_size, nb_fidelities))
    """
    assert X_hf.min() >= 0.0 and X_hf.max() <= 1.0 and torch.all(torch.isfinite(Y_hf))

    # Fidelity parameters
    nb_fidelities = len(models_delta) 
    fidelities = np.arange(nb_fidelities)

    # Create the TR bounds according high fidelity points
    x_center = state.best_xvalue.clone()
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

    dim = X_hf.shape[-1]
    pert = sobol.draw(n_candidates).to(**torchargs)
    pert = tr_lb + pert * (tr_ub - tr_lb)

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, **torchargs) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=torchargs['device'])] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]


    with torch.no_grad() :
        f_samples = []

        if type_of_batch == "parallel" :

            ## samples from the multi-fidelity model
            for i in fidelities :
                f_samples.append(samples_of_fidelity_i(model_low=model_lf,
                                                       models_delta=models_delta,
                                                       rhos=rhos,
                                                       X=X_cand,
                                                       fid=i,
                                                       n_samples=batch_size).unsqueeze(-1))  #[batch, n_candidates, 1]

            f_high_samples = f_samples[-1]  #[batch, n_candidates, 1]

            ## samples from the constraints models [batch, n_candidates, n_constraints]
            constraint_post = constraint_model.posterior(X_cand)
            constraints_samples = constraint_post.rsample(sample_shape=torch.Size([batch_size]))

            ##mask of feasible points
            is_feas = (constraints_samples <= 0).all(dim=-1) #[batch, n_candidates]
            has_feas = is_feas.any(dim=-1) #[batch]

            ##scores
            #if a realization has at least one feasible point, we use the objective value
            #as a score and set all infeasible points to -inf
            #if a realization doesn't have a feasible candidate, we set the score to the 
            #candidate with the lowest total violation
            scores = f_high_samples.squeeze(-1).clone() #[batch, n_candidates]
            scores[~is_feas] = -float("inf")

            #along each batch, if no feasible point, we set the scores to - sum of violations else we keep the scores    
            if not has_feas.all() :
                total_violation = constraints_samples[~has_feas].clamp(min=0).sum(dim=-1) 
                scores[~has_feas] = -total_violation
            
            ##pick the best points according to the type of sampling
            argmax_idx = scores.argmax(dim=-1) #[batch]
            X_next = X_cand[argmax_idx] #[batch, dim]

            ## mean predictions
            means_next = []
            for i in fidelities :
                means_next.append(mean_var_of_fidelity_i(model_low=model_lf,
                                                     models_delta=models_delta,
                                                     rhos=rhos,
                                                     X=X_next,
                                                     fid=i)[0])  #[batch] 
            ## Y predictions
            Y_next = []
            for i in fidelities :
                Y_next.append(f_samples[i][torch.arange(batch_size), argmax_idx, 0]) #[batch]
         
            ## fidelity selection
            best_fidelity, criterion = fidelity_selection(Y_next=Y_next,
                                               means_next=means_next, 
                                               time_costs=time_costs,
                                               fidelities=fidelities)  #[batch, 1]

            ##
            X_next = torch.cat((X_next, best_fidelity), dim=-1) #[batch, dim+1]
        
        elif type_of_batch == "sequential" :
            ## samples from the multi-fidelity model
            for i in fidelities :
                f_samples.append(samples_of_fidelity_i(model_low=model_lf,
                                                       models_delta=models_delta,
                                                       rhos=rhos,
                                                       X=X_cand,
                                                       fid=i,
                                                       n_samples=1).unsqueeze(-1))  #[1, n_candidates, 1]
                
            f_high_samples = f_samples[-1]  #[1, n_candidates, 1]

            ## samples from the constraints models [1, n_candidates, n_constraints]
            constraint_post = constraint_model.posterior(X_cand)
            constraints_samples = constraint_post.rsample(sample_shape=torch.Size([1]))

            ##mask of feasible points
            is_feas = (constraints_samples <= 0).all(dim=-1) #[1, n_candidates]
            has_feas = is_feas.any(dim=-1) #[1]

            ##scores :
            #if a realization has at least one feasible point, we use the objective value
            #as a score and set all infeasible points to -inf
            #if a realization doesn't have a feasible candidate, we set the score to the 
            #candidate with the lowest total violation
            scores = f_high_samples.squeeze(-1).clone() #[1, n_candidates]
            scores[~is_feas] = -float("inf")

            if not has_feas.all() :
                total_violation = constraints_samples[~has_feas].clamp(min=0).sum(dim=-1) #[1, n_candidates]
                scores[~has_feas] = -total_violation

            ##pick the batch_size best points according to the type of sampling
            argmax_idx = scores.topk(batch_size, dim=-1).indices.squeeze(0) #[batch]
            X_next = X_cand[argmax_idx] #[batch, dim]

            ## mean predictions
            means_next = []
            for i in fidelities :
                means_next.append(mean_var_of_fidelity_i(model_low=model_lf,
                                                     models_delta=models_delta,
                                                     rhos=rhos,
                                                     X=X_next,
                                                     fid=i)[0])  #[batch] 

            ## Y predictions
            Y_next = []
            for i in fidelities :
                Y_next.append(f_samples[i][0, argmax_idx, 0]) #[batch]

            ## fidelity selection
            best_fidelity, criterion = fidelity_selection(Y_next=Y_next, 
                                               means_next=means_next, 
                                               time_costs=time_costs, 
                                               fidelities=fidelities)  #[batch, 1]

            ##
            X_next = torch.cat((X_next, best_fidelity), dim=-1) #[batch, dim+1]


        return X_next, criterion


class ScboMfOptimizer:

    def __init__(self, 
                 eval_objectives,
                 constraints,
                 rhos,
                 costs,
                 dim,
                 n_pts,
                 bounds,
                 type_of_batch,
                 type_of_centering,
                 torchargs,
                 restart = False,
                 test = False):
        """ Optimization class for SCBO in order to maximize the objective function. The inputs are in dimension dim+1 where the last dimension is the fidelity dimension (0 : low fidelity, 1 : high fidelity).

        Args:
            eval_objectives (List[callable]): List of functions to evaluate [eval_objective_low,...,eval_objective_high]
            constraints (callable): Function that takes a tensor of shape (n_pts, dim) and returns a tensor of shape (n_pts, n_constraints)
            rhos (List[float]): Correlation parameters between the two fidelities
            costs (List[float]): Time costs of the fidelity functions
            dim (int): Dimension of the problem (without the fidelity dimension)
            n_pts (List[int]): Number of points to sample
            bounds (torch.Tensor): Bounds of the problem of shape (2, dim)
            type_of_batch (str): Type of batch to generate (shoot of batch_size x n_candidates points then select the best ones : "parallel" or  the batch_size best points from n_candidates points : "sequential")
            type_of_centering (str): Type of centering ("evaluated" or "predicted")
            torchargs (dict): Dictionary containing the device on which the tensors will be stored
                                and the dtype (device : str, dtype : torch.dtype)
            restart (bool, optional): If True, the optimizer restarts when the trust region becomes too small. Defaults to False.
            test (bool, optional): If True, the optimizer is in test mode and evaluates best points with high objective in the predicted case. Defaults to False.
        Returns:
            None
           
        """

        self.dim = dim
        self.n_pts = n_pts  # [n_low_fidelity_points, ..., n_high_fidelity_points]
        self.bounds = bounds
        self.torchargs = torchargs
        self.eval_objectives = eval_objectives
        self.rhos = list(rhos)
        self.costs = costs
        self.constraints = constraints

        self.type_of_batch = type_of_batch  # "parallel" or "sequential"
        self.type_of_centering = type_of_centering  # "evaluated" or "predicted"
        self.test = test
        self.restart = restart

        self.trains_X = []
        self.trains_Y = []
        self.C = []

        self.nb_fidelities = len(n_pts)
        self.fidelities = np.arange(self.nb_fidelities)

        self.best_Y = []
        self.best_X = []
        self.best_C = []
        self.generated_Y = []
        self.generated_X = []
        self.generated_C = []
        self.criterions = []
        self.trust_region_lengths = []
        self.gps_info = [[] for _ in range(self.nb_fidelities)]
        self.iterations = []
        self.nb_iter = [[] for _ in range(self.nb_fidelities)]
        self.cost_iter = []
        

        self.paramms = [
            ("dim", self.dim),
            ("bounds", self.bounds.cpu().numpy()),
            ("nb_fidelities", self.nb_fidelities),
            ("n_pts", self.n_pts),
            ("rhos", self.rhos),
            ("costs", self.costs),
            ("type_of_batch", self.type_of_batch),
            ("type_of_centering", self.type_of_centering),
            ("restart", self.restart),
            ("test", self.test),
        ]
        
        assert len(self.n_pts) == len(self.eval_objectives), f"Length of n_pts ({len(self.n_pts)}) must be equal to number of functions ({len(self.eval_objectives)})"
        assert len(self.rhos) == len(self.eval_objectives), f"Length of rhos ({len(self.rhos)}) must be equal to number of functions ({len(self.eval_objectives)})"
        assert len(self.costs) == len(self.eval_objectives), f"Length of costs ({len(self.costs)}) must be equal to number of functions ({len(self.eval_objectives)})"

        assert self.type_of_batch in ["parallel", "sequential"], f"type_of_batch must be 'parallel' or 'sequential', got {self.type_of_batch}"
        assert self.type_of_centering in ["evaluated", "predicted"], f"type_of_centering must be 'evaluated' or 'predicted', got {self.type_of_centering}"
    
    def _print_parameters(self, verbose=True):

        """ Print the parameters of the optimizer.
        Args:
            verbose (bool, optional): If True, print the parameters. Defaults to True.
        """
        if verbose:

            # Format parameters to a better display for arrays
            formatted_params = []
            for key, value in self.paramms:
                if isinstance(value, np.ndarray):
                    if value.ndim == 2:
                        value_str = "[" + ", ".join(str(list(row)) for row in value) + "]"
                else:
                    value_str = str(value)
                formatted_params.append((key, value_str))
            
            # Calculate column widths
            key_width = max(len(k) for k, _ in formatted_params)
            val_width = max(len(str(v)) for _, v in formatted_params)

            title = " Parameters for MF-SCBO Optimization "
            print("\n" + title.center(key_width + val_width + 7, "="))
            print("_" * (key_width + val_width + 7))
            print(f"| {'Parameter'.ljust(key_width)} | {'Value'.ljust(val_width)} |")
            print("-" * (key_width + val_width + 7))

            for key, value in formatted_params:
                print(f"| {key.ljust(key_width)} | {value.ljust(val_width)} |")

            print("-" * (key_width + val_width + 7))

    def _print_optimization_parameters(self, 
                                      max_cost, 
                                      batch_size, 
                                      path, 
                                      verbose=True):
        
        """ Print the optimization parameters.
        Args:
            max_cost (float): Maximum total cost.
            batch_size (int): Batch size.   
            path (str): Path to save the results.
            verbose (bool, optional): If True, print the parameters. Defaults to True.
        """

        if verbose :
            print("Starting MF-SCBO optimization...")
            print(f"Max cost : {max_cost}")
            print(f"Batch size : {batch_size}")
            print(f"Path : {path}")
       
    def _print_iteration(self, 
                        iteration, 
                        current_cost,
                        current_total_cost, 
                        max_cost,
                        current_best_x,
                        current_best_y,
                        length_tr,
                        is_feasible,
                        smallest_total_violation,
                        generated_X,
                        generated_Y,
                        criterion,
                        generated_fidelities,
                        verbose=True): 
        """ Print the results of the current iteration.
        Args:
            iteration (int): Current iteration number.
            current_cost (float): Cost of the current iteration.
            current_total_cost (float): Current total cost.
            max_cost (float): Maximum total cost.
            current_best_x (np.ndarray): Current best x point.
            current_best_y (float): Current best y value.
            length_tr (float): Current trust region length.
            is_feasible (bool): Whether the current best point is feasible.
            smallest_total_violation (float): Smallest total violation of the constraints.
            generated_X (np.ndarray): Generated points in the current iteration.
            generated_Y (np.ndarray): Generated y values in the current iteration.
            generated_fidelities (list): Generated fidelities in the current iteration.
            verbose (bool, optional): If True, print the iteration results. Defaults to True.
        """    

        if verbose :

            title = f" Iteration {iteration} : cost/max_cost = {current_total_cost}/{max_cost} "
            print("\n" + "-" * (len(title) + 15))
            print(title.center(15 + len(title), " "))
            print("-" * (len(title) + 15))

            print(f"Feasible : {is_feasible}")
            print(f"Smallest total violation : {smallest_total_violation:.3e}")
            print(f"Best x : {current_best_x}")
            print(f"Best y : {current_best_y:.3e}")
            print(f"Trust region length : {length_tr:.3e}")
            print(f"Cost of iteration : {current_cost}")
            print(f"Total points : {sum([sum(nb_iter_fid_i) for nb_iter_fid_i in self.nb_iter])}") #sum of all nb_iter

            print("Generated x :")
            print(generated_X)
            print("Generated y :")
            print(generated_Y)
            print("Associated criterion :" )
            print(criterion)
            print("Generated fidelities :")
            print(generated_fidelities)
           
    def _print_gp_parameters(self, model, fidelity, verbose=True):
        """ Print the parameters of the GP model.
        Args:
            model (botorch.models.SingleTaskGP or DeltaGPModel): GP model to print the parameters of.
            fidelity (int): Fidelity level of the model.
            verbose (bool, optional): If True, print the parameters. Defaults to True.
        """
        gp_info = get_gp_info(model, fidelity)
        noise = gp_info['noise']
        noise_bounds = gp_info['noise_bounds']
        length_scale = gp_info['length_scale']
        length_scale_bounds = gp_info['length_scale_bounds']  
        rho = gp_info['rho']
        rho_bounds = gp_info['rho_bounds']  
        #parameter | value | bounds
        if verbose :
            params = []

            params.append((
                "Length scale",
                length_scale,
                f"({length_scale_bounds[0]:.2e}, {length_scale_bounds[1]:.2e})"
            ))
            params.append((
                "Noise",
                f"{noise:.2e}",
                f"({noise_bounds[0]:.2e}, {noise_bounds[1]:.2e})"
            ))
            if fidelity > 0:
                params.append((
                    "Rho",
                    f"{rho:.3e}",
                    f"({rho_bounds[0]:.2e}, {rho_bounds[1]:.2e})"
                ))

            # Column widths 
            col1_w = max(len(p[0]) for p in params)      # Parameter
            col2_w = max(len(str(p[1])) for p in params) # Value
            col3_w = max(len(str(p[2])) for p in params) # Bounds

            total_w = col1_w + col2_w + col3_w + 10

            
            title = f" GP Model Parameters (fidelity {fidelity}) "
            print("\n" + title.center(total_w, "="))
            print("_" * total_w)
            print(f"| {'Parameter'.ljust(col1_w)} | "
                f"{'Value'.ljust(col2_w)} | "
                f"{'Bounds'.ljust(col3_w)} |")
            print("-" * total_w)

            for p, v, b in params:
                print(f"| {p.ljust(col1_w)} | "
                    f"{str(v).ljust(col2_w)} | "
                    f"{str(b).ljust(col3_w)} |")

            print("-" * total_w)

    def _create_dir_and_log(self, path):
        """ Create the directory and log file for the optimization.

        Args:
            path (str): Path to the directory.
        """
        os.makedirs(path, exist_ok=True)
        log_file = open(path + "/MFSCBO_log.txt", "w")
        sys.stdout = Print_log_and_console(sys.stdout, log_file)
    
    def _close_log(self):
        """ Close the log file. """
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    def _update_data(self,
                    iteration,
                    trains_X,
                    trains_Y,
                    trains_C,
                    state_length,
                    state_best_xvalue,
                    state_best_value,
                    state_best_constraint_values,
                    criterion,
                    model_low,
                    models_delta): 
        """ Update the data to be saved.

        Args:
            iteration (int): Current iteration number.
            trains_X (List[torch.Tensor]): List of training points for each fidelity level.
            trains_Y (List[torch.Tensor]): List of training values for each fidelity level.
            trains_C (List[torch.Tensor]): List of constraint values for each fidelity level.
            state_length (float): Current trust region length.
            state_best_xvalue (torch.Tensor): Current best x point.
            state_best_value (float): Current best y value.
            state_best_constraint_values (torch.Tensor): Current best constraint values.
            criterion (torch.Tensor): Criterion values for the current iteration.
            model_low (botorch.models.SingleTaskGP): Fitted low fidelity GP model.
            models_delta (List[botorch.models.SingleTaskGP]): Fitted delta GP models.
        """

        
        trains_X_unnormalized = []
        for fid in self.fidelities :
            trains_X_unnormalized.append(unnormalize(trains_X[fid][:,:-1], self.bounds))
            trains_X_unnormalized[fid] = torch.cat((trains_X_unnormalized[fid], fid * torch.ones(trains_X_unnormalized[fid].shape[0], 1, **self.torchargs)), dim=1)
        self.generated_X.append(torch.cat(trains_X_unnormalized, dim=0).cpu().detach())
        self.generated_Y.append(torch.cat(trains_Y, dim=0).cpu().detach())
        self.generated_C.append(torch.cat(trains_C, dim=0).cpu().detach())
        self.trust_region_lengths.append(state_length)
        self.best_X.append(unnormalize(state_best_xvalue.cpu().detach(), self.bounds))
        self.best_Y.append(state_best_value)
        self.best_C.append(state_best_constraint_values.cpu().detach())
        self.iterations.append(iteration)
        self.criterions.append(criterion.cpu().detach())

        for fid in self.fidelities :
            if fid == 0 and model_low is not None :
                    gp_info = get_gp_info(model_low, fid)
                    self.gps_info[fid].append(gp_info)
            elif fid > 0 and models_delta is not None :
                    gp_info = get_gp_info(models_delta[fid], fid)
                    self.gps_info[fid].append(gp_info)

    def _save(self, 
             gps_info,
             generated_X,
             generated_Y,
             generated_C,
             criterions,
             trust_region_lengths,
             best_X,
             best_Y,
             best_C,
             iterations,
             nb_iter,
             cost_iter,
             path ): 
        """ Save the optimization results.

        Args:
            gps_info (List[dict]): List of GP model information for each fidelity level.
            generated_X (List[torch.Tensor]): List of generated points for each iteration.
            generated_Y (List[torch.Tensor]): List of generated values for each iteration.
            generated_C (List[torch.Tensor]): List of generated constraint values for each iteration.
            criterions (List[torch.Tensor]): List of criterion values for each iteration.
            trust_region_lengths (List[float]): List of trust region lengths for each iteration.
            best_X (List[torch.Tensor]): List of best points (according best_Y) for each iteration.
            best_Y (List[torch.Tensor]): List of best values for each iteration.
            best_C (List[torch.Tensor]): List of constraints (according best_Y) values for each iteration.
            iterations (List[int]): List of iteration numbers.
            nb_iter (List[List[int]]): List of number of iterations for each fidelity level.
            cost_iter (List[float]): List of total costs for each iteration.
            path (str): Path to save the results.
        """
        
        np.savez(path+"/MFSCBO_data.npz",
                gps_info = np.array(gps_info, dtype=object),
                generated_X = np.array(generated_X, dtype=object),
                generated_Y = np.array(generated_Y, dtype=object),
                generated_C = np.array(generated_C, dtype=object),
                criterions = np.array(criterions, dtype=object),
                trust_region_lengths = np.array(trust_region_lengths, dtype=object),
                best_X= np.array(best_X, dtype=object),
                best_Y= np.array(best_Y, dtype=object),
                best_C= np.array(best_C, dtype=object),
                iterations = np.array(iterations),
                nb_iter = np.array(nb_iter),
                cost_iter = np.array(cost_iter))
        
    def _save_info(self,
                  type_of_centering,
                  type_of_batch,
                  batch_size,
                  max_cost,
                  restart,
                  test,
                  path ) :
        
        """ Save the optimization information.
        Args:
            type_of_centering (str): Type of centering ("evaluated" or "predicted").
            type_of_batch (str): Type of batch ("parallel" or "sequential").
            batch_size (int): Batch size.
            max_cost (float): Maximum total cost.
            restart (bool): Whether the optimizer restarts when the trust region becomes too small.
            test (bool): Whether the optimizer is in test mode and evaluates best points with high objective in the predicted case.
            path (str): Path to save the results.
        """
        
        info = {
            "type_of_centering": type_of_centering,
            "type_of_batch": type_of_batch,
            "batch_size": batch_size,
            "costs": self.costs,
            "restart": restart,
            "test": test,
            "max_cost": max_cost
        }

        np.savez(path+"/MFSCBO_info.npz", **info)

    def initialize(self, added_X=None, added_Y=None, added_C=None):
        """Initialize the optimization by sampling initial points.

        Args:
            added_X (List[torch.Tensor], optional): Additional points (normalized). 
                                              Defaults to None.
            added_Y (List[torch.Tensor], optional): Additional objective values. 
                                              Defaults to None.
            added_C (List[torch.Tensor], optional): Additional constraint values. 
                                              Defaults to None.
        """

        if added_X is not None:
            self.trains_X = [x.clone().detach() for x in added_X]
        else :
            self.trains_X = get_initial_points_mf(self.dim, self.n_pts, self.torchargs)

        if added_Y is not None:
            self.trains_Y = [y.clone().detach() for y in added_Y]
        else :
            self.trains_Y = []
            for fid in self.fidelities :
                Y_fid = self.eval_objectives[fid](self.trains_X[fid][:,:-1])
                self.trains_Y.append(Y_fid)

        if added_C is not None:
            self.C = [c.clone().detach() for c in added_C]
        else :
            self.C = []
            for fid in self.fidelities :
                C_fid = self.constraints(self.trains_X[fid][:,:-1])
                self.C.append(C_fid)
        

    def optimize(self, 
                 max_cost=100, 
                 batch_size=4, 
                 gp_parameters = { "objective" : {"noise_interval": (1e-6, 1e-3),
                                                 "matern_nu": 2.5,
                                                 "lengthscale_interval": (0.005, 4.0)},
                                  "constraints" : {"noise_interval": (1e-6, 1e-3),
                                                 "matern_nu": 2.5,
                                                 "lengthscale_interval": (0.005, 4.0)},
                                                 "is_binary": False},
                 path = "MFSCBO", verbose=True):
        """ Optimize the objective function using MF-SCBO.

        Args:
            max_cost (int, optional): max cost = n_pts[0]*time_costs[0] + ... + n_pts[S]*time_costs[S]. Defaults to 100.
            batch_size (int, optional): Batch size for the optimization. Defaults to 4.
            gp_parameters (dict, optional): Dictionary containing the GP hyperparameters. If different parameters for the different constraints,
                                            the dictionay should be of the form {"objective" : {...}, "constraints" : {0 : {...}, 1 :{...}, ..., n_constraints-1 : {...}}}.
                                            Defaults to {"objective" : {"noise_interval": (1e-6, 1e-3),
                                                                        "matern_nu": 2.5,
                                                                        "lengthscale_interval": (0.005, 4.0)},
                                                        "constraints" : {"noise_interval": (1e-6, 1e-3),
                                                                        "matern_nu": 2.5,  
                                                                        "lengthscale_interval": (0.005, 4.0)},
                                                                        "is_binary": False}.
            path (str, optional): Path to save the results. Defaults to "MFSCBO".
            verbose (bool, optional): If True, print the optimization progress. Defaults to True.
        
        Returns:
            (torch.Tensor, torch.Tensor): The best X and Y values found during optimization.
        """

        self._create_dir_and_log(path)
        self._print_optimization_parameters(max_cost=max_cost, 
                                          batch_size=batch_size, 
                                          path=path, 
                                          verbose=verbose)
        self._print_parameters(verbose=verbose)
        self._save_info(type_of_centering=self.type_of_centering,
                       type_of_batch=self.type_of_batch,
                    batch_size=batch_size,
                       max_cost=max_cost,
                       restart=self.restart,
                       test=self.test,
                       path=path)

        noise_interval = gp_parameters["objective"]["noise_interval"]
        martern_nu = gp_parameters["objective"]["matern_nu"]
        lengthscale_interval = gp_parameters["objective"]["lengthscale_interval"]

    
        n_candidates=min(5000, max(2000, 200 * self.dim))
        sobol = SobolEngine(self.dim, scramble=True,)
        
        
        state = ScboState(self.dim, batch_size, self.C[0].shape[-1],self.torchargs)
        state = update_state(state=state, 
                             Y_next_hf=self.trains_Y[-1],
                             X_next_hf=self.trains_X[-1][:,:-1],
                             C_next_hf=self.C[-1], 
                             model_low=None,
                             models_delta=None,
                             rhos=None,
                             type_of_centering="evaluated") #TODO
        

      
        k = 0
        self._update_data(iteration=k,
                          trains_X=self.trains_X,
                          trains_Y=self.trains_Y,
                          trains_C=self.C,
                          state_length=state.length,
                          state_best_xvalue=state.best_xvalue,
                          state_best_value=state.best_value,
                          state_best_constraint_values=state.best_constraint_values,
                          criterion=torch.zeros((1, self.nb_fidelities), **self.torchargs),
                          model_low=None,
                          models_delta=None)
        
        for fid in self.fidelities :
            self.nb_iter[fid].append(self.n_pts[fid])
        cost = torch.sum(torch.tensor(self.n_pts) * torch.tensor(self.costs))
        self.cost_iter.append(cost.item())


        self._save(gps_info=self.gps_info,
                  generated_X=self.generated_X,
                  generated_Y=self.generated_Y,
                  generated_C=self.generated_C,
                  criterions=self.criterions,
                  trust_region_lengths=self.trust_region_lengths,
                  best_X=self.best_X,
                  best_Y=self.best_Y,
                  best_C=self.best_C,
                  iterations=self.iterations,
                  nb_iter=self.nb_iter,
                  cost_iter=self.cost_iter,
                  path=path)

        #store high fidelity in case of type_of_centering = "evaluated"
        Y_high_inter = torch.zeros((1,1), **self.torchargs)
        X_high_inter = torch.zeros((1, self.trains_X[0].shape[-1]), **self.torchargs)
        C_high_inter = torch.zeros((1, self.C[0].shape[-1]), **self.torchargs)


        while not state.restart_triggered and (cost < max_cost):
            gc.collect()
            torch.cuda.empty_cache()


            ##Fit of the multi-fidelity model
            model_low = get_fitted_model(self.trains_X[0][:, :-1], self.trains_Y[0], noise_interval, martern_nu, lengthscale_interval)

            ##Create delta models
            models_delta = [None]
            for fid in self.fidelities :
                if fid != 0 :
                    X_fid_i = self.trains_X[fid][:, :-1]
                    X_fid_im1 = self.trains_X[fid - 1][:, :-1]
                    Y_fid_i = self.trains_Y[fid]
                    Y_fid_im1 = self.trains_Y[fid - 1]

                    #get indices of X_fid_i that are not in X_fid_im1
                    mask_out_of_fid_i, mask_in_of_fid_i, mask_in_of_fid_im1 = out_and_in(X_fid_i, X_fid_im1)

                    #get Y_fid_im1
                    Y_fid_im1_in_fid_i = Y_fid_im1[mask_in_of_fid_im1]
                    Y_fid_im1_out_fid_i_pred = mean_var_of_fidelity_i(model_low, models_delta, self.rhos, X_fid_i[mask_out_of_fid_i], fid - 1)[0]
                    Y_fid_im1_out_fid_i_pred = Y_fid_im1_out_fid_i_pred[:,None]
                   
                    #concatenate the two parts to get Y_fid_im1 as high fidelity points
                    virtual_Y_fid_im1 = torch.cat((Y_fid_im1_in_fid_i, Y_fid_im1_out_fid_i_pred), dim=0)
                    virtual_X_fid_im1 = torch.cat((X_fid_i[mask_in_of_fid_i], X_fid_i[mask_out_of_fid_i]), dim=0)
                    Y_fid_i_ordered = torch.cat((Y_fid_i[mask_in_of_fid_i], Y_fid_i[mask_out_of_fid_i]), dim=0)
                    
                    #delta model fit
                    model_delta = DeltaGPModel(virtual_X_fid_im1, #all in X_fid_i points
                                               Y_fid_i_ordered,
                                               virtual_Y_fid_im1,
                                               covar_module=gpytorch.kernels.ScaleKernel(
                                                    gpytorch.kernels.MaternKernel(
                                                        nu=2.5,
                                                        ard_num_dims=self.dim,
                                                        lengthscale_constraint=gpytorch.constraints.Interval(0.005, 4.0),
                                                    )
                                                ),
                                                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                                                   noise_constraint=gpytorch.constraints.Interval(1e-6, 1e-3)
                                               ),
                                               torchargs=self.torchargs
                                               ).to(**self.torchargs)
                    
                    #fit the model
                    fit_model_delta(model_delta, virtual_X_fid_im1)
                  
                    self.rhos[fid] = model_delta.rho.item()
                    models_delta.append(model_delta) 
                
    
            #Fit of the constraints models
            gp_parameters_constraints = gp_parameters["constraints"]
            
            if 0 in gp_parameters_constraints : #if 0 is a key then, different parameters for each constraint so use index i else use the same parameters for all constraints
                list_constraints_model = [get_fitted_model(torch.cat(self.trains_X, dim=0)[:,:-1], torch.cat(self.C, dim=0)[:, i][:, None], **gp_parameters_constraints[i]) for i in range(self.C[0].shape[1])]
            else :
                list_constraints_model = [get_fitted_model(torch.cat(self.trains_X, dim=0)[:,:-1], torch.cat(self.C, dim=0)[:, i][:, None], **gp_parameters_constraints) for i in range(self.C[0].shape[1])]

            with gpytorch.settings.max_cholesky_size(float("inf")):

                X_next, criterion = generate_batch_mf(
                    type_of_batch=self.type_of_batch,
                    state=state,
                    model_lf=model_low,
                    models_delta=models_delta,
                    rhos=self.rhos,
                    time_costs = self.costs,
                    X_hf=self.trains_X[-1][:,:-1] if not state.still_violated else torch.cat(self.trains_X, dim=0)[:,:-1], #if we are still violated, we consider all points as high fidelity because constraints are the same for all fidelities
                    Y_hf=self.trains_Y[-1] if not state.still_violated else torch.cat(self.trains_Y, dim=0), #if we are still violated, we consider all points as high fidelity because constraints are the same for all fidelities
                    C_hf=self.C[-1] if not state.still_violated else torch.cat(self.C, dim=0), #if we are still violated, we consider all points as high fidelity because constraints are the same for all fidelities
                    batch_size=batch_size,
                    n_candidates=n_candidates,
                    constraint_model=ModelListGP(*list_constraints_model),
                    sobol=sobol,
                    torchargs=self.torchargs
                )


            #count number fidelity points in the batch
            n_fid_points = []
            for fid in self.fidelities :
                n_fid_points.append(torch.sum(X_next[:,-1]==fid).item())

            #for all points in X_next (where X_next[:,-1] is the fidelity associated) evaluate the given fidelity function
            # Y_next should me same order as X_next
            Y_next = torch.zeros((X_next.shape[0], 1), **self.torchargs)
            for i in range(X_next.shape[0]) :
                fid = int(X_next[i,-1].item())
                Y_next[i] = self.eval_objectives[fid](X_next[i,:-1][None,:])
            

            #constraints for all points in X_next
            C_next = self.constraints(X_next[:,:-1])

            #intermediate storage of high fidelity points and constraints
            Y_high_inter = torch.cat((Y_high_inter, Y_next[X_next[:,-1]==self.fidelities[-1]]), dim=0)
            X_high_inter = torch.cat((X_high_inter, X_next[X_next[:,-1]==self.fidelities[-1], :]), dim=0)
            C_high_inter = torch.cat((C_high_inter, C_next[X_next[:,-1]==self.fidelities[-1]]), dim=0)

            #update state
            if state.still_violated :
                print("Warning : infeasible, we update the state with all fidelity points because constraints are the same for all fidelities")
                state = update_state(state=state, 
                                     Y_next_hf=Y_next,
                                     X_next_hf=X_next[:,:-1], 
                                     C_next_hf=C_next,
                                     model_low=None,
                                     models_delta=None,
                                     rhos=None,
                                     type_of_centering="evaluated") #TODO
            
            else :

                if self.type_of_centering == "evaluated" : #we update the state with evaluated high fidelity points
                    if Y_high_inter.shape[0] >= batch_size + 1 : #+1 because of the initial zero
                        state = update_state(state=state, 
                                             Y_next_hf=Y_high_inter[1:], 
                                             X_next_hf=X_high_inter[1:, :-1],
                                             C_next_hf=C_high_inter[1:],
                                             model_low=model_low,
                                             models_delta= models_delta,
                                             rhos=self.rhos,
                                             type_of_centering=self.type_of_centering)
                        #reset high fidelity storage
                        Y_high_inter = torch.zeros((1,1), **self.torchargs)
                        X_high_inter = torch.zeros((1, self.trains_X[0].shape[-1]), **self.torchargs)
                        C_high_inter = torch.zeros((1, self.C[0].shape[-1]), **self.torchargs)
                
                elif self.type_of_centering == "predicted" : #we update the state with high fidelity predictions of the X_next points
                    Y_high_pred = mean_var_of_fidelity_i(model_low, models_delta, self.rhos, X_next[:,:-1], self.fidelities[-1])[0]
                        

                    #/!\ TODO : use analytic formula for LOO criterion instead of recomputing the model with leave one out
                    state_inter = update_state(state=copy.deepcopy(state), 
                                Y_next_hf=Y_high_pred, 
                                X_next_hf=X_next[:,:-1],
                                C_next_hf=C_next, 
                                model_low=model_low,
                                models_delta=models_delta,
                                rhos=self.rhos,
                                type_of_centering=self.type_of_centering)
                    #compute leave one out of the center of the trust region according the fidelity level of the center of the trust region
                    center_tr_region = state_inter.best_xvalue
                    #compare state.best_xvalue with the points in X_next to find the index of the center of the trust region
                    index_center_tr_region = torch.where((X_next[:,:-1] == center_tr_region).all(dim=1))[0]
                    if index_center_tr_region.shape[0] != 0 :
                        fid_center_tr_region = int(X_next[index_center_tr_region[0], -1].item())
                        train_X_loo = [None for _ in self.fidelities]
                        train_Y_loo = [None for _ in self.fidelities]
                        for fid in self.fidelities :
                                train_X_loo[fid] = self.trains_X[fid]
                                train_Y_loo[fid] = self.trains_Y[fid]
                        #add X_next and Y_next except the center of the trust region to the training data
                        for i in range(X_next.shape[0]) :
                            if i != index_center_tr_region[0] :
                                fid = int(X_next[i,-1].item())
                                train_X_loo[fid] = torch.cat((train_X_loo[fid], X_next[i,:][None,:]), dim=0)
                                train_Y_loo[fid] = torch.cat((train_Y_loo[fid], Y_next[i,:][None,:]), dim=0)
                    
                        #fit gps and constraints 
                        model_low_loo = get_fitted_model(train_X_loo[0][:, :-1], train_Y_loo[0], noise_interval, martern_nu, lengthscale_interval)
                        models_delta_loo = [None]
                        rhos_loo = [None for _ in self.fidelities]
                        for fid in self.fidelities :
                            if fid != 0 :
                                X_fid_i_loo = train_X_loo[fid][:, :-1]
                                X_fid_im1_loo = train_X_loo[fid - 1][:, :-1]
                                Y_fid_i_loo = train_Y_loo[fid]
                                Y_fid_im1_loo = train_Y_loo[fid - 1]

                                #get indices of X_fid_i that are not in X_fid_im1
                                mask_out_of_fid_i_loo, mask_in_of_fid_i_loo, mask_in_of_fid_im1_loo = out_and_in(X_fid_i_loo, X_fid_im1_loo)

                                #get Y_fid_im1
                                Y_fid_im1_in_fid_i_loo = Y_fid_im1_loo[mask_in_of_fid_im1_loo]
                                Y_fid_im1_out_fid_i_pred_loo = mean_var_of_fidelity_i(model_low_loo, models_delta_loo, self.rhos, X_fid_i_loo[mask_out_of_fid_i_loo], fid - 1)[0]
                                Y_fid_im1_out_fid_i_pred_loo = Y_fid_im1_out_fid_i_pred_loo[:,None]

                                #concatenate the two parts to get Y_fid_im1 as high fidelity points
                                virtual_Y_fid_im1_loo = torch.cat((Y_fid_im1_in_fid_i_loo, Y_fid_im1_out_fid_i_pred_loo), dim=0)
                                virtual_X_fid_im1_loo = torch.cat((X_fid_i_loo[mask_in_of_fid_i_loo], X_fid_i_loo[mask_out_of_fid_i_loo]), dim=0)
                                Y_fid_i_ordered_loo = torch.cat((Y_fid_i_loo[mask_in_of_fid_i_loo], Y_fid_i_loo[mask_out_of_fid_i_loo]), dim=0)

                                #delta model fit
                                model_delta_loo = DeltaGPModel(virtual_X_fid_im1_loo, #all in X_fid_i points
                                                        Y_fid_i_ordered_loo,
                                                        virtual_Y_fid_im1_loo,
                                                        covar_module=gpytorch.kernels.ScaleKernel(
                                                                gpytorch.kernels.MaternKernel(
                                                                    nu=2.5,
                                                                    ard_num_dims=self.dim,
                                                                    lengthscale_constraint=gpytorch.constraints.Interval(0.005, 4.0),
                                                                )
                                                            ),
                                                            likelihood=gpytorch.likelihoods.GaussianLikelihood(
                                                            noise_constraint=gpytorch.constraints.Interval(1e-6, 1e-3)
                                                        )
                                                        , torchargs=self.torchargs
                                                        )
                                fit_model_delta(model_delta_loo, virtual_X_fid_im1_loo)

                                rhos_loo[fid] = model_delta_loo.rho.item()
                                models_delta_loo.append(model_delta_loo)
                        
                        #error loo betwenn predicted value of models_loo and the true value of the center of the trust region
                        pred_center_tr_region, var_center_tr_region = mean_var_of_fidelity_i(model_low_loo, models_delta_loo, rhos_loo, state.best_xvalue[None,:], fid_center_tr_region)#compliqué de fixer car il faut que le modèle soit fit, ce qui se fait au début de la boucle
                        conf_inter_95_loo = 1.96 * torch.sqrt(var_center_tr_region).item()
                        error_loo = torch.abs(pred_center_tr_region - self.eval_objectives[fid_center_tr_region](state.best_xvalue[None,:])[0]).item()
                        print(f"================= error : {error_loo} =================")
                        print(f"================= conf_inter_95 : {conf_inter_95_loo} =================")
                        if error_loo > conf_inter_95_loo:
                            print("ADD HIGH FID POINT")
                            #we evaluate the true value of the center with the high fidelity function and update the state 
                            true_value_center = self.eval_objectives[-1](state.best_xvalue[None,:])[0]
                            #augment X_next, Y_next, C_next  
                            #X_next with high fidelity point
                            X_next = torch.cat((X_next, torch.cat((state.best_xvalue[None,:], torch.tensor([[self.fidelities[-1]]], **self.torchargs)), dim=1)), dim=0)
                            Y_next = torch.cat((Y_next, true_value_center[None,:]), dim=0)
                            C_next = torch.cat((C_next, self.constraints(state.best_xvalue[None,:])), dim=0)
                            
                            #now pred
                            Y_high_pred = mean_var_of_fidelity_i(model_low, models_delta, self.rhos, X_next[:,:-1], self.fidelities[-1])[0]
                            state = update_state(state=state, 
                                                 Y_next_hf=Y_high_pred, 
                                                 X_next_hf=X_next[:,:-1],
                                                 C_next_hf=C_next,
                                                 model_low=model_low,
                                                 models_delta=models_delta,
                                                 rhos=self.rhos,
                                                 type_of_centering=self.type_of_centering)
                            
                            #augment n_fid_points for the high fidelity point
                            n_fid_points[-1] += 1
                    else :
                        state = copy.deepcopy(state_inter)
                            
                


            # concatenate the new points to the training data
            for i in range(X_next.shape[0]) :
                fid = int(X_next[i,-1].item())
                self.trains_X[fid] = torch.cat((self.trains_X[fid], X_next[i,:][None,:]), dim=0)

            for i in range(Y_next.shape[0]) :
                fid = int(X_next[i,-1].item())
                self.trains_Y[fid] = torch.cat((self.trains_Y[fid], Y_next[i,:][None,:]), dim=0)

            for i in range(C_next.shape[0]) :
                fid = int(X_next[i,-1].item())
                self.C[fid] = torch.cat((self.C[fid], C_next[i,:][None,:]), dim=0)

   
            k += 1
            if self.test and self.type_of_centering == "predicted" :
                state_best_value = self.eval_objectives[-1](state.best_xvalue[None,:]).item()
            else :
                state_best_value = state.best_value
            self._update_data(iteration=k,
                             trains_X=self.trains_X,
                             trains_Y=self.trains_Y,
                             trains_C=self.C,
                             state_length=state.length,
                             state_best_xvalue=state.best_xvalue,
                             state_best_value=state_best_value,
                             state_best_constraint_values=state.best_constraint_values,
                             criterion=criterion,
                             model_low=model_low,
                             models_delta=models_delta)

            for fid in self.fidelities :
                self.nb_iter[fid].append(n_fid_points[fid])
            
            cost = 0
            for fid in self.fidelities :
                cost += torch.sum(torch.tensor(self.nb_iter[fid])) * self.costs[fid]
            self.cost_iter.append(cost.item())


            self._print_iteration(iteration=k,
                                 current_cost=sum([n_fid_points[fid]*self.costs[fid] for fid in self.fidelities]),
                                 current_total_cost=cost.item(),
                                 max_cost=max_cost,
                                 current_best_x=self.best_X[-1].numpy(),
                                 current_best_y=self.best_Y[-1],
                                 length_tr=state.length,
                                 is_feasible=(state.best_constraint_values <= 0).all().item(),
                                 smallest_total_violation=state.best_constraint_values.clamp(min=0).sum().item(),
                                 generated_X=self.generated_X[-1].numpy(),
                                 generated_Y=self.generated_Y[-1].numpy(),
                                 criterion=criterion.cpu().numpy(),
                                 generated_fidelities=n_fid_points,
                                 verbose=verbose)
   
            for fid in self.fidelities :
                if fid == 0 :
                    self._print_gp_parameters(model = model_low, 
                                             fidelity=fid, 
                                             verbose=verbose)
                else :
                    self._print_gp_parameters(model = models_delta[fid], 
                                             fidelity=fid, 
                                             verbose=verbose)
            
            self._save(gps_info=self.gps_info,
                      generated_X=self.generated_X,
                      generated_Y=self.generated_Y,
                      generated_C=self.generated_C,
                      criterions=self.criterions,
                      trust_region_lengths=self.trust_region_lengths,
                      best_X=self.best_X,
                      best_Y=self.best_Y,
                      best_C=self.best_C,
                      iterations=self.iterations,
                      nb_iter=self.nb_iter,
                      cost_iter=self.cost_iter,
                      path=path)
                    
            
            if self.restart and state.restart_triggered :
                state.restart_triggered = False
                state.length = 0.8

        self._close_log()

        return self.best_X[-1], self.best_Y[-1]

