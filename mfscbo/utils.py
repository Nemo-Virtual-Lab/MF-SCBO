import sys
import torch
from botorch.utils.transforms import unnormalize

COLORS = ["#6aab5d", "#635edf", "#d96260","#efc266"]
MARKERS = ['o', 's', '^', 'v']


class Print_log_and_console(object):
    """Class to print both in a file and in the console."""
    def __init__(self, *files):
        """Initialize the class with the given files.
            Args:
                files: File objects where to write the output.
        """
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()
    def close(self): #close files except stdout and stderr
        for f in self.files:
            if f not in (sys.__stdout__, sys.__stderr__):
                f.close()

def eval_fun_unnormalized(x, fun, bounds):
    """Evaluate a function on normalized points.

    Args:
        x (torch.Tensor): Points to evaluate of shape (n_pts, dim)
        fun (callable): Function to evaluate
        bounds (torch.Tensor): Bounds of shape (2, dim)

    Returns:
        torch.Tensor : Evaluated de-normalized points
    """
    return fun(unnormalize(x, bounds))

def get_gp_info(model, fidelity):
    """Get information about the GP model.
    Args:
        model (botorch.models.SingleTaskGP or DeltaGPModel): GP model
        fidelity (int): Fidelity level of the model
    Returns:
        dict: Dictionary containing GP information : "length_scale", "length_scale_bounds", "noise", "noise_bounds", "rho", "rho_bounds".
    """

    noise = model.likelihood.noise.item()
    length_scale = model.covar_module.base_kernel.lengthscale.detach()
    length_scale = length_scale.squeeze()
    length_scale = "[" + ", ".join(f"{ls:.2e}" for ls in length_scale.cpu().numpy()) + "]"
    
    length_scale_bounds = model.covar_module.base_kernel.raw_lengthscale_constraint.lower_bound, model.covar_module.base_kernel.raw_lengthscale_constraint.upper_bound
    length_scale_bounds = (length_scale_bounds[0].item(), length_scale_bounds[1].item())
    noise_bounds = model.likelihood.noise_covar.raw_noise_constraint
    noise_bounds = (noise_bounds.lower_bound.item(), noise_bounds.upper_bound.item())

    if fidelity == 0 :
        rho = None
        rho_bounds = None
    else : 
        rho = model.rho.item()
        rho_bounds = (model.raw_rho_constraint.lower_bound, model.raw_rho_constraint.upper_bound)
        rho_bounds = (rho_bounds[0].item(), rho_bounds[1].item())

    gp_info = {
        "length_scale": length_scale,
        "length_scale_bounds": length_scale_bounds,
        "noise": noise,
        "noise_bounds": noise_bounds,
        "rho": rho,
        "rho_bounds": rho_bounds
    }   

    return gp_info


def ensure_list(val, default, num_fidelities):
    """Ensure that the input value is a list of length num_fidelities.
    Args:
        val (any or list or tuple): Input value.
        default (any): Default value to use if val is None.
        num_fidelities (int): Number of fidelities.
    Returns:
        list: List of length num_fidelities.
    """
    if val is None:
        return default
    if not isinstance(val, (list, tuple)):
        return [val] * num_fidelities
    return val

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
