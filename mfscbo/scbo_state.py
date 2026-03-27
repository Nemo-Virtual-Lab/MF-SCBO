from dataclasses import dataclass
import math
import torch
from torch import Tensor
from mfscbo.mfgp import mean_of_fidelity_i


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Original code from BoTorch : https://botorch.org/docs/tutorials/scalable_constrained_bo/
# Some modifications have been made to adapt the code to  our needs.
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@dataclass
class ScboState:
    dim: int
    batch_size: int
    num_constraints: int 
    torchargs: dict
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3 # original TuRBO value
    best_value: float = -float("inf")
    best_xvalue: Tensor = None  # Note : Post-initialized
    best_constraint_values: Tensor = None # Note : Post-initialized
    best_ind: int = 0
    still_violated: bool = True
    restart_triggered: bool = False

    """ State class for SCBO optimization. """

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))
        self.best_constraint_values = torch.ones(self.num_constraints, **self.torchargs) * torch.inf

def update_tr_length(state: ScboState):
    """ Update the trust region length according to the success and failure counters.
    Args:
        state (ScboState): Current state of the optimization
    Returns:
        ScboState: Updated state of the optimization
    """
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state

def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point.
    Args:
        Y (torch.Tensor): Objective values of shape (n_pts, 1)
        C (torch.Tensor): Constraint values of shape (n_pts, n_constraints)
    Returns:
        int: Index of the best point in the batch
    """
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()

def update_state(state, 
                 Y_next_hf, 
                 X_next_hf, 
                 C_next_hf, 
                 model_low, 
                 models_delta, 
                 rhos, 
                 type_of_centering):
    """ Update the state of the optimization.

    Args:
        state (ScboState): Current state of the optimization
        Y_next_hf (torch.Tensor): New objective high fidelity values of shape (n_pts, 1)
        X_next_hf (torch.Tensor): New high fidelity points of shape (n_pts, dim)
        C_next_hf (torch.Tensor): New constraint values of shape (n_pts, n_constraints) for high fidelity points
        model_low (botorch.models.SingleTaskGP): Fitted low fidelity GP model
        models_delta (List[botorch.models.SingleTaskGP]): Fitted delta GP models
        rhos (List[float]): Correlation parameters between fidelities
        type_of_centering (str): Type of centering to use ("evaluated" or "predicted")


    Returns:
        ScboState: Updated state of the optimization
    """
  
    #if no high fidelity in the batch, we do not update the state
    if Y_next_hf is None :
        return state
    

    #Pick the best point from the batch
    best_ind = get_best_index_for_batch(Y=Y_next_hf, C=C_next_hf)

    y_next = Y_next_hf[best_ind]
    x_next = X_next_hf[best_ind]
    c_next = C_next_hf[best_ind]
    
    if type_of_centering == "predicted":
       
        #update actual center (state.best_value) with the new predicted value associated to model_high_fidelity
        state.best_value = mean_of_fidelity_i(model_low=model_low,
                                              models_delta=models_delta,
                                              rhos=rhos,
                                              X=state.best_xvalue[None, :],
                                              fid=len(models_delta)-1).item() #because [fid_0, fid_1, ..., fid_S] so S = len(models_delta)-1

    if (c_next <= 0).all():
        state.still_violated = False
        # At least one new candidate is feasible
        improvement_threshold = state.best_value + 1e-6 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_xvalue = x_next.clone().detach()
            state.best_constraint_values = c_next
            state.best_ind = best_ind 
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_xvalue = x_next.clone().detach()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    # Update the length of the trust region according to the success and failure counters
    state = update_tr_length(state)
    return state
