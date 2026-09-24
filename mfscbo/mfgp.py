
import copy
import torch
import gpytorch
from torch.nn import Parameter
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_mll_scipy
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from botorch.models.gpytorch import GPyTorchModel
import warnings
from gpytorch.utils.cholesky import psd_safe_cholesky
from mfscbo.utils import out_and_in



class DeltaGPModel(gpytorch.models.ExactGP):
    """GP model for the delta function in multi-fidelity optimization (in order to learn rho)."""

    def __init__(self, X_fid_i, Y_fid_i, Y_fid_im1, covar_module, likelihood, torchargs):
        """Initialize the DeltaGPModel.
        Args:
            X_fid_i (torch.Tensor): Input points at fidelity i of shape (n_i, dim)
            Y_fid_i (torch.Tensor): Output points at fidelity i of shape (n_i, 1)
            Y_fid_im1 (torch.Tensor): Output points at fidelity i-1 of shape (n_im1, 1)
            covar_module (gpytorch.kernels.Kernel): Covariance module for the GP model
            likelihood (gpytorch.likelihoods.Likelihood): Likelihood for the GP model
            torchargs (dict): Dictionary containing the device on which the tensors will be stored
                                and the dtype (device : str, dtype : torch.dtype)
        """
        super().__init__(X_fid_i, Y_fid_i, likelihood)

        self.Y_fid_im1 = Y_fid_im1.clone().detach()
        self.Y_fid_i = Y_fid_i.clone().detach()
        self.likelihood = likelihood
        self.torchargs = torchargs

        # rho 
        self.raw_rho = Parameter(torch.tensor(1.0, **torchargs))
        self.register_parameter("raw_rho", self.raw_rho)
        self.register_constraint("raw_rho", gpytorch.constraints.Interval(0., 2.0))

  
        self.mean_module = gpytorch.means.ConstantMean().to(**torchargs)
        self.covar_module = covar_module.to(**torchargs)
        self.condition_on_residuals()

    @property
    def rho(self):
        """Get the current value of rho."""
        return self.raw_rho_constraint.transform(self.raw_rho)

    def forward(self, x):
        """Compute the GP output at points x.
        Args:
            x (torch.Tensor): Input points of shape (n_pts, dim)
        Returns:
            gpytorch.distributions.MultivariateNormal: GP output distribution
        """
        delta_mean = self.mean_module(x)
        delta_covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(delta_mean, delta_covar)
    
    def check_rho(self, threshold=1e-3):
        """Warn if rho is too close to zero."""

        rho = self.rho.detach().item()

        if rho < threshold:
            print(f"rho is very close to 0: rho = {rho:.3e}")
         

    def delta_targets(self):
        """Compute the delta targets for training, i.e., Y_fid_i - rho * Y_fid_im1.
        Returns:
            torch.Tensor: Delta targets of shape (n_i, 1)
        """
        return self.Y_fid_i - self.rho * self.Y_fid_im1

    def condition_on_residuals(self):
        """Set the training targets of the GP to the delta targets (with the current rho), so that the posterior is conditioned on them.
        To call again each time rho, Y_fid_i or Y_fid_im1 change.
        """
        self.set_train_data(targets=self.delta_targets().squeeze(-1).detach(), strict=False)

    def posterior(self, X_new, likelihood=True):
        """Compute the posterior predictive distribution at X_new.

        Args:
            X_new (torch.Tensor): New input points of shape (n_new_pts, dim)
            likelihood (bool, optional): If True, returns the posterior through the likelihood (including observation noise).
                                          If False, returns the latent function posterior. Defaults to True.

        Returns:
            gpytorch.distributions.MultivariateNormal: Posterior predictive distribution
        """
        self.eval()
        if likelihood:
            self.likelihood.eval()
        with torch.no_grad():
            mvn = self(X_new) #posterior conditioned on the delta targets
            if likelihood:
                mvn = self.likelihood(mvn)
        return mvn

def mean_var_of_fidelity_i(model_low, models_delta, rhos, X, fid):
    """Compute the mean and variance of the fidelity i function at points X.

    Args:
        model_low (botorch.models.SingleTaskGP): Low fidelity GP model
        models_delta (List[botorch.models.SingleTaskGP]): List of delta GP models for each fidelity : [None, delta_1, delta_2, ..., delta_i]
        rhos (List[float]): List of rho values for each fidelity : [1, rho_1, rho_2, ..., rho_i]
        X (torch.Tensor): Input points of shape (n_pts, dim)
        fid (int): Fidelity level (0 for low fidelity, S for high fidelity)

    Returns:
        torch.Tensor: Mean predictions of shape (n_pts)
    """

    if fid == 0 :
        return model_low.posterior(X).mean.squeeze(-1), model_low.posterior(X).variance.squeeze(-1)
    else :
        mean = model_low.posterior(X).mean.squeeze(-1)
        var = model_low.posterior(X).variance.squeeze(-1)
        for i in range(1, fid + 1):
            rho_i = rhos[i]
            model_delta_i = models_delta[i]
            mean_delta_i = model_delta_i.posterior(X).mean.squeeze(-1)
            mean = rho_i * mean + mean_delta_i
            var_delta_i = model_delta_i.posterior(X).variance.squeeze(-1)
            var = (rho_i ** 2) * var + var_delta_i
        return mean, var


def samples_of_fidelity_i(model_low, models_delta, rhos, X, fid, n_samples):
    """Compute samples of the fidelity i function at points X.

    Args:
        model_low (botorch.models.SingleTaskGP): Low fidelity GP model
        models_delta (List[botorch.models.SingleTaskGP]): List of delta GP models for each fidelity : [None, delta_1, delta_2, ..., delta_i]
        rhos (List[float]): List of rho values for each fidelity : [1, rho_1, rho_2, ..., rho_i]
        X (torch.Tensor): Input points of shape (n_pts, dim)
        fid (int): Fidelity level (0 for low fidelity, S for high fidelity)
        n_samples (int): Number of samples to draw
    Returns:
        torch.Tensor: Sampled predictions of shape (n_samples, n_pts)
    """

    if fid == 0 :
        return model_low.posterior(X).rsample(sample_shape=torch.Size([n_samples])).squeeze(-1)
    else :
        samples = model_low.posterior(X).rsample(sample_shape=torch.Size([n_samples])).squeeze(-1)
        for i in range(1, fid + 1):
            rho_i = rhos[i]
            model_delta_i = models_delta[i]
            delta_i_samples = model_delta_i.posterior(X).rsample(sample_shape=torch.Size([n_samples])).squeeze(-1)
            samples = rho_i * samples + delta_i_samples
        return samples

def loo_function(K, y, mean_constant):
    """leave-one-out mean and variance of a GP with fixed hyperparameters.

    Args:
        K (torch.Tensor): Covariance matrix of the observations, noise included, of shape (n_pts, n_pts)
        y (torch.Tensor): Observations of shape (n_pts,)
        mean_constant (torch.Tensor or float): Constant prior mean of the GP

    Returns:
        (torch.Tensor, torch.Tensor): (LOO means of shape (n_pts,), LOO variances of shape (n_pts,)).
                                      The variance is the one of a noisy observation because K contains the noise.
    """
    L = psd_safe_cholesky(K) 
    alpha = torch.cholesky_solve((y - mean_constant)[:, None], L).squeeze(-1) #K^-1 (y - m), in case mean_constant not null
    L_inv = torch.linalg.solve_triangular(L, torch.eye(K.shape[0], dtype=K.dtype, device=K.device), upper=False)
    diag_K_inv = (L_inv ** 2).sum(dim=0) #(K^-1)_ii = sum_k (L^-1)_ki^2
    return y - alpha / diag_K_inv, 1.0 / diag_K_inv

def virtual_lower_data(X_fid_i, X_fid_im1, Y_fid_i, Y_fid_im1, lower_mean):
    """Lower fidelity values at the points of fidelity i.The points are ordered as [nested points ; non-nested points].

    Args:
        X_fid_i (torch.Tensor): Points at fidelity i of shape (n_i, dim)
        X_fid_im1 (torch.Tensor): Points at fidelity i-1 of shape (n_im1, dim)
        Y_fid_i (torch.Tensor): Values at fidelity i of shape (n_i, 1)
        Y_fid_im1 (torch.Tensor): Values at fidelity i-1 of shape (n_im1, 1)
        lower_mean (callable): Predicted mean of fidelity i-1, takes points of shape (n, dim) and returns shape (n,)

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): (ordered X_fid_i of shape (n_i, dim), ordered Y_fid_i of shape (n_i, 1),
                                                                  virtual Y_fid_im1 of shape (n_i, 1), order : indices in X_fid_i of shape (n_i,))
    """
    mask_out_of_fid_i, mask_in_of_fid_i, mask_in_of_fid_im1 = out_and_in(X_fid_i, X_fid_im1)
    order = torch.cat((torch.where(mask_in_of_fid_i)[0], torch.where(mask_out_of_fid_i)[0]))
    virtual_Y_fid_im1 = torch.cat((Y_fid_im1[mask_in_of_fid_im1], lower_mean(X_fid_i[mask_out_of_fid_i])[:, None]), dim=0)
    return X_fid_i[order], Y_fid_i[order], virtual_Y_fid_im1, order

def condition_mf_model(model_low, models_delta, rhos, X_list, Y_list, up_to_fid):
    """Condition copies of the multi-fidelity GP models on new data, the hyperparameters, rhos and standardization fixed.

    Args:
        model_low (botorch.models.SingleTaskGP): Fitted low fidelity GP model
        models_delta (List[DeltaGPModel]): Fitted delta GP models : [None, delta_1, delta_2, ..., delta_S]
        rhos (List[float]): List of rho values for each fidelity : [1, rho_1, rho_2, ..., rho_S]
        X_list (List[torch.Tensor]): Points per fidelity of shape (n_i, dim+1) (last column is the fidelity)
        Y_list (List[torch.Tensor]): Values per fidelity of shape (n_i, 1)
        up_to_fid (int): Highest fidelity level to condition

    Returns:
        (botorch.models.SingleTaskGP, List[DeltaGPModel]): (conditioned low fidelity model, [None, conditioned delta_1, ..., conditioned delta_up_to_fid])
    """
    model_low_c = copy.deepcopy(model_low).eval()
    model_low_c.set_train_data(X_list[0][:, :-1], model_low_c.outcome_transform(Y_list[0])[0].squeeze(-1), strict=False)

    models_delta_c = [None]
    for fid in range(1, up_to_fid + 1):
        lower_mean = lambda X: mean_var_of_fidelity_i(model_low_c, models_delta_c, rhos, X, fid - 1)[0]
        virtual_X, Y_fid_i_ordered, virtual_Y_fid_im1, _ = virtual_lower_data(X_list[fid][:, :-1], X_list[fid - 1][:, :-1],
                                                                              Y_list[fid], Y_list[fid - 1], lower_mean)
        model_delta_c = copy.deepcopy(models_delta[fid])
        model_delta_c.Y_fid_i = Y_fid_i_ordered.clone().detach()
        model_delta_c.Y_fid_im1 = virtual_Y_fid_im1.clone().detach()
        model_delta_c.set_train_data(inputs=virtual_X, strict=False)
        model_delta_c.condition_on_residuals()
        models_delta_c.append(model_delta_c)

    return model_low_c, models_delta_c

def analytical_loo(model_low, models_delta, rhos, trains_X, trains_Y, X_batch, Y_batch, idx, fid):
    """Analytical LOO mean and variance at the point idx of the batch, for its fidelity fid.
    The full model is trained on trains ∪ batch with the hyperparameters and rhos of the fitted models held fixed,
    then the point idx is removed.

    Args:
        model_low (botorch.models.SingleTaskGP): Fitted low fidelity GP model
        models_delta (List[DeltaGPModel]): Fitted delta GP models : [None, delta_1, delta_2, ..., delta_S]
        rhos (List[float]): List of rho values for each fidelity : [1, rho_1, rho_2, ..., rho_S]
        trains_X (List[torch.Tensor]): Training points per fidelity of shape (n_i, dim+1) (last column is the fidelity)
        trains_Y (List[torch.Tensor]): Training values per fidelity of shape (n_i, 1)
        X_batch (torch.Tensor): Batch points of shape (batch_size, dim+1) (last column is the fidelity)
        Y_batch (torch.Tensor): Batch values of shape (batch_size, 1)
        idx (int): Index in the batch of the point to leave out
        fid (int): Fidelity level of the point idx

    Returns:
        (torch.Tensor, torch.Tensor): (LOO mean, LOO variance) at the point idx
    """
    # add the batch to the training data, for all fidelities
    X_full = [torch.cat((trains_X[i], X_batch[X_batch[:, -1] == i]), dim=0) for i in range(fid + 1)]
    Y_full = [torch.cat((trains_Y[i], Y_batch[X_batch[:, -1] == i]), dim=0) for i in range(fid + 1)]
    pos = trains_X[fid].shape[0] + int((X_batch[:idx, -1] == fid).sum().item())

    model_low.eval()

    if fid == 0 :
        X_0 = X_full[0][:, :-1]
        y_std = model_low.outcome_transform(Y_full[0])[0].squeeze(-1) #standardization of the fitted model (fixed)
        noise = model_low.likelihood.noise
        K = model_low.covar_module(X_0).to_dense() + noise * torch.eye(X_0.shape[0], dtype=X_0.dtype, device=X_0.device)
        mean_loo, var_loo = loo_function(K, y_std, model_low.mean_module.constant)
        stdv = model_low.outcome_transform.stdvs.squeeze()
        mean = model_low.outcome_transform.means.squeeze() + stdv * mean_loo[pos]
        var = stdv ** 2 * (var_loo[pos] - noise.squeeze()) 
        return mean, var

    # lower fidelities conditioned on data and hyperparameters fixed 
    model_low_full, models_delta_full = condition_mf_model(model_low, models_delta, rhos, X_full, Y_full, fid - 1)
    lower_mean = lambda X: mean_var_of_fidelity_i(model_low_full, models_delta_full, rhos, X, fid - 1)[0]
    virtual_X, Y_fid_ordered, virtual_Y_fid_im1, order = virtual_lower_data(X_full[fid][:, :-1], X_full[fid - 1][:, :-1],
                                                                            Y_full[fid], Y_full[fid - 1], lower_mean)
    pos = int(torch.where(order == pos)[0].item())

    # LOO of the delta model on the residuals r = y - rho * y_virtual
    rho = rhos[fid]
    model_delta = models_delta[fid]
    r = (Y_fid_ordered - rho * virtual_Y_fid_im1).squeeze(-1)
    K = model_delta.covar_module(virtual_X).to_dense() + model_delta.likelihood.noise * torch.eye(virtual_X.shape[0], dtype=virtual_X.dtype, device=virtual_X.device)
    mean_delta_loo, var_delta_loo = loo_function(K, r, model_delta.mean_module.constant)

    mean_low, var_low = mean_var_of_fidelity_i(model_low_full, models_delta_full, rhos, virtual_X[pos][None, :], fid - 1)
    return rho * mean_low.squeeze() + mean_delta_loo[pos], rho ** 2 * var_low.squeeze() + var_delta_loo[pos]

def get_fitted_model(X, Y, noise_interval, matern_nu, lengthscale_interval, is_binary=False):
    """Fit a GP model to data (X, Y).

    Args:
        X (torch.Tensor): Input data of shape (n_pts, dim)
        Y (torch.Tensor): Output data of shape (n_pts, 1)
        noise_interval (Tuple[float, float]): Interval for the noise variance hyperparameter (min, max)
        matern_nu (float): Smoothness parameter for the Matern kernel (e.g., 0.5 for Matern-1/2, 1.5 for Matern-3/2, 2.5 for Matern-5/2)
        lengthscale_interval (Tuple[float, float]): Interval for the lengthscale hyperparameters (min, max)
        is_binary (bool, optional): If True, use a binary likelihood. Defaults to False.

    Returns:
        botorch.models.SingleTaskGP: Fitted GP model
    """

    if is_binary :
        model = Binary_GPModel(
            X,
            Y,
            covar_module=gpytorch.kernels.scale_kernel.ScaleKernel(gpytorch.kernels.RBFKernel()),
        )
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=X.shape[0])
        model.double()
        mll.double()

    else : 
        model = SingleTaskGP(
            X,
            Y,
            covar_module=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=matern_nu,
                    ard_num_dims=X.shape[1],
                    lengthscale_constraint=gpytorch.constraints.Interval(*lengthscale_interval),
                )
            ),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(*noise_interval)
            ),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(float("inf")):
        fit_gpytorch_mll(mll)

    return model

def fit_model_delta(model_delta, X) :
    """Fit the delta GP model by optimizing the marginal log likelihood with fit_gpytorch_mll_scipy to ensure constraints.

    Args:
        model_delta (DeltaGPModel): Delta GP model to fit
        X (torch.Tensor): Input data of shape (n_pts, dim)
    """ 
    # Marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        model_delta.likelihood,
        model_delta
    )

    # Prepare model for training 
    model_delta.train()
    model_delta.likelihood.train()


    def mll_closure():
        # clear previous grads because : 
        # call 1 : p.grad = grad1
        # call 2 : p.grad = grad1 + grad2
        # ...
        for p in mll.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # recompute delta targets (rho changes)
        delta_Y = model_delta.delta_targets().squeeze()

        # loss
        output = model_delta(X)
        loss = -mll(output, delta_Y)

        # compute gradients
        grad_tensors = torch.autograd.grad(
            loss,
            tuple(mll.parameters()),
            allow_unused=True, #return None for parameters that do not affect the loss
            retain_graph=False,
            create_graph=False
        )

        return loss, grad_tensors
    
    #scipy constrained optimization
    fit_gpytorch_mll_scipy(
        mll,
        closure=mll_closure,
        method="L-BFGS-B",      # uses bounds automatically
        options={"disp": False}
    )

    # condition the posterior on the delta targets with the fitted rho
    model_delta.condition_on_residuals()


class Binary_GPModel(ApproximateGP,GPyTorchModel) : #from https://botorch.org/docs/notebooks_community/clf_constrained_bo/
    """GP model for binary classification in constrained Bayesian optimization."""
    def __init__(self, X, Y, covar_module) : 
        """Initialize the Binary_GPModel.

        Args:
            X (torch.Tensor): Input data of shape (n_pts, dim)
            Y (torch.Tensor): Binary output data of shape (n_pts, 1) with values in {0, 1}
            covar_module (gpytorch.kernels.Kernel): Covariance module for the GP model
        """
        
        assert Y.shape[-1] == 1, f"Y must be of shape [batch_size, 1], but got {Y.shape}"
        
        self.train_inputs = (X,)
        self.train_targets = Y[:,0] 

        variational_distribution = CholeskyVariationalDistribution(X.size(0))
        variational_strategy = VariationalStrategy(
            self, X, variational_distribution
        )
        super(Binary_GPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    
    def forward(self, x) :
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

