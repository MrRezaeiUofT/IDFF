import math
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from torchcfm.models.models import MLP
from torchcfm.utils import *# sample_8gaussians, sample_checkerboard, sample_moons,sample_spirals,sample_pinwheel
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchdiffeq import odeint
import torchsde
from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances
import itertools
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
# Setup
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

def log_prob_xt_given_x1_dimwise(x_t, x_1, x_0, t, sigma0_sq):
    """
    Compute log p_t(x_t | x_1) per dimension: returns shape (batch_size, dim)

    Args:
        x_t, x_1, x_0: Tensors of shape (batch_size, dim)
        t: Tensor of shape (batch_size,) or scalar
        sigma0_sq: scalar

    Returns:
        log_probs: Tensor of shape (batch_size, dim)
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=x_t.dtype, device=x_t.device)
    if t.ndim == 1:
        t = t[:, None]  # shape (batch_size, 1)

    sigma0_sq = torch.tensor(sigma0_sq, dtype=x_t.dtype, device=x_t.device)

    mu_t = t * x_1 + (1 - t) * x_0
    sigma2 = sigma0_sq * t * (1 - t)+1e-4  # shape (batch_size, 1)

    diff = x_t - mu_t

    log_probs = -0.5 * (torch.log(2 * torch.pi * sigma2) + (diff ** 2) / sigma2)
    return log_probs  # shape: (batch_size, dim)


savedir = "sample_8gaussians/"
os.makedirs(savedir, exist_ok=True)

# Hyperparameters
sigma = 0.2
dim = 4  # 2 for x_hat 
batch_size = 256


# Define model
class CustomNetModel(torch.nn.Module):
    def __init__(self, base_model, sigma=0.2, gamma1=0.5, gamma2=0.5):
        super(CustomNetModel, self).__init__()
        self.base_model = base_model
        self.sigma = sigma
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def forward(self, t, xt):
        xt = xt.requires_grad_(True)

        sigma_t2 = (self.sigma * torch.sqrt(t * (1 - t)))**2

        if t.item() == 0.0:
            self.x0 = xt

        t_input = t * torch.ones((xt.shape[0], 1), device=xt.device)
        model_input = torch.cat([xt,xt,t_input], dim=-1)
        predictions = self.base_model(model_input)

        if t == 0:

            return predictions[:,:2] - xt  # Replace self.x0 with first-order gradient
        elif t == 1:
            return (predictions[:,:2] - xt) / 1e-2
        else:
            

            return (
                torch.sqrt(1 - (self.gamma1+self.gamma2)*sigma_t2) * (predictions[:,:2] - xt) / ((1 - t))
                + 0.5 * (2 * self.gamma1 - 1) * predictions[:,2:] * sigma_t2  # First-order gradient term
                + self.gamma2* torch.ones_like(xt) * sigma_t2 #/ (self.sigma ** 2)  # Second-order gradient term
            )

use_cuda = False
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
# Instantiate base model and optimizer
base_model = MLP(dim=dim, w=256, time_varying=True).to(device)
optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-3)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0)

# Training Phase
best_loss = float('inf')
best_model_path = os.path.join(savedir, "best_IDFF_model.pt")
start = time.time()
Training = True

if Training:
    for k in range(70000):
        optimizer.zero_grad()
        true_samples = sample_8gaussians(batch_size).to(device)
        # Sample data
        x0 = torch.randn_like(true_samples).to(device)
        t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, true_samples, return_noise=True)

        # Compute outputs
        outs = base_model(torch.cat([xt,xt, t[:, None]], dim=-1))
        vt=outs[:,:dim//2]
        st=outs[:,dim//2:]

        sigma_t = sigma * torch.sqrt(t * (1 - t))
        lambda_t = 2 * sigma_t / (sigma ** 2 + 1e-8)
        grads = log_prob_xt_given_x1_dimwise(xt, FM.x1, FM.x0,t,sigma) 
        # print(grads.shape)
        # Compute losses
        flow_loss = torch.mean(((1/(1e-2+1-t[:, None]))*(vt - FM.x1)) ** 2)
        # flow_loss = torch.mean((vt - FM.x1) ** 2)
        grad_loss = torch.mean(((st - grads)) ** 2)
        loss = flow_loss +grad_loss

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1)  # new
        optimizer.step()

        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(base_model.state_dict(), best_model_path)

        # Print progress and evaluation
        if (k + 1) % 1000 == 0:
            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f}, flow_loss {flow_loss.item():0.3f} , grad_loss {grad_loss.item():0.3f} ,time {(end - start):0.2f}s (Best Loss {best_loss:0.3f})")
            start = end

    print("Training complete.")

# Testing Phase
print("\nLoading best model and running tests...")

# Load best model
test_model = base_model
test_model.load_state_dict(torch.load(best_model_path,map_location=torch.device('cpu')))
test_model.eval()
sample_size=batch_size = 2048
true_samples = sample_8gaussians(batch_size).to(device).detach().cpu().numpy()
x0 = torch.randn((sample_size, 2)).to(device) /1.5
nfe=3

def plot_kde_evolution_grid(test_model, sample_size=512, n_samples_to_overlay=5, 
                            nfe=10, gamma_settings=[(0.,0.), (1,0), (0,.5), (1.,.5)],
                            custom_models=None,
                            save_name="kde_evolution_grid_arrows"):
    """
    Plot KDE evolution over time for different gamma settings and models (rotated layout).
    Time steps are now rows, and models are columns.
    """
    t_span = torch.linspace(0, 1, nfe+1, device=device)
    x0_base = x0

    true_kde = gaussian_kde(true_samples.T)
    xgrid, ygrid = np.meshgrid(np.linspace((true_samples).min()-1, (true_samples).max()+1, 50), 
                               np.linspace((true_samples).min()-1, (true_samples).max()+1, 50))
    grid_points = np.vstack([xgrid.ravel(), ygrid.ravel()])
    z_true = true_kde(grid_points).reshape(xgrid.shape)

    total_cols = len(gamma_settings) + (len(custom_models) if custom_models else 0)
    fig, axs = plt.subplots(nfe+1, total_cols, figsize=(3.5 * total_cols, 3.5 * (nfe+1)))

    if nfe+1 == 1 or total_cols == 1:
        axs = np.atleast_2d(axs)

    col_index = 0

    # Custom models (Diffusion / CFM)
    if custom_models:
        for label, model in custom_models:
            traj = odeint(model, x0_base, t_span, rtol=1e-5, atol=1e-5, method='rk4')
            traj_np = traj.cpu().detach().numpy()

            for row in range(nfe+1):
                ax = axs[row, col_index]
                kde_gen = gaussian_kde(traj_np[row].T)
                z_gen = kde_gen(grid_points).reshape(xgrid.shape)

                ax.contourf(xgrid, ygrid, z_gen, levels=40, cmap='Reds', alpha=0.7)
                ax.contour(xgrid, ygrid, z_true, levels=8, colors='black', linestyles='dashed', linewidths=1)
                ax.contour(xgrid, ygrid, z_gen, levels=6, colors='blue', linestyles='solid', linewidths=0.8)

                # Select overlay samples far apart
                overlay_indices = [0]
                overlay_points = traj_np[0]
                for _ in range(1, n_samples_to_overlay):
                    dists = np.min([np.linalg.norm(overlay_points - overlay_points[i], axis=1) for i in overlay_indices], axis=0)
                    dists[overlay_indices] = -np.inf
                    next_idx = np.argmax(dists)
                    overlay_indices.append(next_idx)

                for i in overlay_indices:
                    for step in range(row):
                        x_start, y_start = traj_np[step, i]
                        dx = traj_np[step + 1, i, 0] - x_start
                        dy = traj_np[step + 1, i, 1] - y_start
                        ax.arrow(x_start, y_start, dx, dy,
                                 color='blue', alpha=0.8, width=0.02, head_width=.5, head_length=.5,
                                 length_includes_head=True)
                    ax.scatter(traj_np[0, i, 0], traj_np[0, i, 1], color='blue', s=40, marker='o', edgecolor='k')
                    ax.scatter(traj_np[row, i, 0], traj_np[row, i, 1], color='blue', s=60, marker='*', edgecolor='k')

                ax.set_xlim((true_samples).min()-1, (true_samples).max()+1)
                ax.set_ylim((true_samples).min()-1, (true_samples).max()+1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(label, fontsize=11)
                if col_index == 0:
                    ax.set_ylabel(f"t={t_span[row].item():.2f}", fontsize=11)

            col_index += 1

    # IDFF variants
    for gamma1, gamma2 in gamma_settings:
        model = CustomNetModel(test_model, sigma=sigma, gamma1=gamma1, gamma2=gamma2).to(device)
        traj = odeint(model, x0_base, t_span, rtol=1e-5, atol=1e-5, method='rk4')
        traj_np = traj.cpu().detach().numpy()

        for row in range(nfe+1):
            ax = axs[row, col_index]
            kde_gen = gaussian_kde(traj_np[row].T)
            z_gen = kde_gen(grid_points).reshape(xgrid.shape)

            ax.contourf(xgrid, ygrid, z_gen, levels=40, cmap='Reds', alpha=0.7)
            ax.contour(xgrid, ygrid, z_true, levels=8, colors='black', linestyles='dashed', linewidths=1)
            ax.contour(xgrid, ygrid, z_gen, levels=6, colors='blue', linestyles='solid', linewidths=0.8)

            overlay_indices = [0]
            overlay_points = traj_np[0]
            for _ in range(1, n_samples_to_overlay):
                dists = np.min([np.linalg.norm(overlay_points - overlay_points[i], axis=1) for i in overlay_indices], axis=0)
                dists[overlay_indices] = -np.inf
                next_idx = np.argmax(dists)
                overlay_indices.append(next_idx)

            for i in overlay_indices:
                for step in range(row):
                    x_start, y_start = traj_np[step, i]
                    dx = traj_np[step + 1, i, 0] - x_start
                    dy = traj_np[step + 1, i, 1] - y_start
                    ax.arrow(x_start, y_start, dx, dy,
                             color='blue', alpha=0.8, width=0.02, head_width=.5, head_length=.5,
                             length_includes_head=True)
                ax.scatter(traj_np[0, i, 0], traj_np[0, i, 1], color='blue', s=40, marker='o', edgecolor='k')
                ax.scatter(traj_np[row, i, 0], traj_np[row, i, 1], color='blue', s=60, marker='*', edgecolor='k')

            ax.set_xlim((true_samples).min()-1, (true_samples).max()+1)
            ax.set_ylim((true_samples).min()-1, (true_samples).max()+1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$\gamma_1={gamma1}, \gamma_2={gamma2}$", fontsize=11)
            if col_index == 0:
                ax.set_ylabel(f"t={t_span[row].item():.2f}", fontsize=11)

        col_index += 1

    plt.tight_layout()
    plt.savefig(savedir + f"{save_name}_rotated.png")
    plt.savefig(savedir + f"{save_name}_rotated.svg", format='svg')
    plt.show()

def compute_mmd_rbf(x, y, sigma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two distributions x and y using an RBF kernel.
    Inputs:
        x: torch.Tensor of shape [N, D] (true samples)
        y: torch.Tensor of shape [M, D] (generated samples)
    Returns:
        scalar MMD value
    """
    def rbf_kernel(a, b, sigma):
        dists = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dists / (2 * sigma ** 2))

    Kxx = rbf_kernel(x, x, sigma)
    Kyy = rbf_kernel(y, y, sigma)
    Kxy = rbf_kernel(x, y, sigma)

    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd.item()
    

def find_best_gammas_mmd(gamma_range=[0.0, 0.5, 1.0, 2.0], sample_size=4096, nfe=20, sigma_kernel=1.0):
    t_span = torch.linspace(0, 1, nfe+1, device=device)
    

    best_mmd = float('inf')
    best_gamma = (None, None)
    results = []

    for gamma1, gamma2 in itertools.product(gamma_range, repeat=2):
        model = CustomNetModel(test_model, sigma=sigma, gamma1=gamma1, gamma2=gamma2).to(device)
        traj = odeint(model, x0, t_span, rtol=1e-5, atol=1e-5, method='rk4')
        pred_samples = traj[-1]

        mmd = compute_mmd_rbf(torch.tensor(true_samples), pred_samples, sigma=sigma_kernel)
        results.append(((gamma1, gamma2), mmd))

        print(f"γ₁={gamma1}, γ₂={gamma2} -> MMD: {mmd:.5f}")
        if mmd < best_mmd:
            best_mmd = mmd
            best_gamma = (gamma1, gamma2)

    print(f"\nBest gamma combination: γ₁={best_gamma[0]}, γ₂={best_gamma[1]} with MMD={best_mmd:.5f}")
    return best_gamma, results

best_gamma, all_results = find_best_gammas_mmd(gamma_range=list(np.arange(-5.0, 5, 0.5)) , sample_size=4096, nfe=nfe)
gamma_settings=[(0,0), (best_gamma[0],0),(0, best_gamma[1]), best_gamma]



class CFMModel(torch.nn.Module):
    def __init__(self, base_model, sigma=0.2, gamma1=0.5, gamma2=0.5):
        super(CFMModel, self).__init__()
        self.base_model = base_model
        self.sigma = sigma
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def forward(self, t, xt):
        xt = xt.requires_grad_(True)

        sigma_t2 = (self.sigma * torch.sqrt(t * (1 - t)))**2

        if t.item() == 0.0:
            self.x0 = xt

        t_input = t * torch.ones((xt.shape[0], 1), device=xt.device)
        model_input = torch.cat([xt,xt,t_input], dim=-1)
        predictions = self.base_model(model_input)

        if t == 0:
           self.x0= xt  # Replace self.x0 with first-order gradient
        return predictions[:,:2]-self.x0

    

cfm_model =  CFMModel(test_model, sigma=0, gamma1=0, gamma2=0).to(device)

custom_models = [

    ("CFM", cfm_model),
]

plot_kde_evolution_grid(test_model, 
                        sample_size=4098, 
                        n_samples_to_overlay=12,
                        nfe=nfe,
                        gamma_settings=[(0,0), (best_gamma[0],0), best_gamma],
                        custom_models=custom_models,
                        save_name="kde_with_baselines")


