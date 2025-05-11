import os
import time
import pickle
import sys
sys.path.append('../')
import numpy as np
import torch
import torchsde
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torchcfm.models.models import MLP_Embedding
from scipy.stats import entropy
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher, pad_t_like_x
def compute_kl_divergence(true_samples, generated_samples, bandwidth=1.0, epsilon=1e-10):
    """
    Compute the KL Divergence between two distributions using Kernel Density Estimation (KDE),
    with smoothing to avoid zero probabilities.

    Args:
        true_samples (np.ndarray): Samples from the true distribution, shape (N, D).
        generated_samples (np.ndarray): Samples from the generated distribution, shape (M, D).
        bandwidth (float): Bandwidth parameter for KDE.
        epsilon (float): Small value added for smoothing.

    Returns:
        float: KL Divergence value.
    """
    kde_true = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(true_samples)
    kde_gen = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(generated_samples)

    # Evaluate densities on a common grid
    common_samples = np.vstack([true_samples, generated_samples])
    log_density_true = kde_true.score_samples(common_samples)
    log_density_gen = kde_gen.score_samples(common_samples)

    # Convert log densities to probabilities
    p = np.exp(log_density_true)
    q = np.exp(log_density_gen)

    # Apply smoothing
    p = (p + epsilon) / (p + epsilon).sum()
    q = (q + epsilon) / (q + epsilon).sum()

    # Compute KL Divergence
    kl_div = entropy(p, q)
    return kl_div

def compute_mmd(x, y, sigma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two distributions x and y using a Gaussian kernel.
    
    Args:
        x (torch.Tensor): Samples from the first distribution, shape (N, D).
        y (torch.Tensor): Samples from the second distribution, shape (M, D).
        sigma (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        float: MMD value.
    """
    def rbf_kernel(x1, x2, sigma):
        """
        Compute the Gaussian RBF kernel between x1 and x2.
        
        Args:
            x1 (torch.Tensor): First set of samples, shape (N, D).
            x2 (torch.Tensor): Second set of samples, shape (M, D).
            sigma (float): Bandwidth parameter for the kernel.

        Returns:
            torch.Tensor: Pairwise kernel matrix, shape (N, M).
        """
        pairwise_dists = torch.cdist(x1, x2, p=2)  # Compute pairwise distances
        return torch.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
    
    # Ensure the tensors have the same dtype
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    
    x_kernel = rbf_kernel(x, x, sigma)  # Kernel within x
    y_kernel = rbf_kernel(y, y, sigma)  # Kernel within y
    xy_kernel = rbf_kernel(x, y, sigma)  # Kernel between x and y

    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()  # MMD formula
    return mmd.item()

class TrajectoryGenerator:
    def __init__(self, dataset_name, sigma,nfe, batch_size, num_trajectories, save_dir, model_name, use_cuda=False):
        self.dataset_name = dataset_name
        self.sigma = sigma
        self.batch_size = batch_size
        self.num_trajectories = num_trajectories
        self.save_dir = os.path.join(save_dir, dataset_name)
        self.model_name = model_name
        self.rgn_thr = 30
        self.NFE=nfe
        self.num_steps = 50000
        self.vis_step = 10000
        self.device = torch.device("cuda" if use_cuda else "cpu")

        os.makedirs(self.save_dir, exist_ok=True)
        self._load_dataset()

    def _load_dataset(self):
        dataset_path = f"data/{self.dataset_name}_dataset.p"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        with open(dataset_path, 'rb') as file:
            data = pickle.load(file)
        orig_trajectories = data['angles'][1].reshape([data['sim_length'], -1])
        self.dataset = torch.tensor(orig_trajectories).squeeze().float().to(self.device)
        self.DataDim = orig_trajectories.shape[-1]
        self.max_length = self.dataset.shape[0]

    def train_model(self):
        model = MLP_Embedding(dim=self.DataDim, w=128, time_embed=self.max_length, time_varying=True).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())

        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=self.sigma)
        
        best_loss = float('inf')
        best_model_path = os.path.join(self.save_dir, f"{self.model_name}.pt")

        for step in range(self.num_steps):
            optimizer.zero_grad()
            index_sel = torch.randint(1, self.max_length, (self.batch_size,)).to(self.device)
            x0 = self.dataset[index_sel - 1]
            x1 = self.dataset[index_sel]

            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)

            # xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            vt = model(torch.cat([xt, t[:, None]], dim=-1), index_sel)

            loss = (
                torch.mean((vt - x1) ** 2)
                + torch.mean((vt - (x1 + 360)) ** 2)
                + torch.mean((vt - (x1 - 360)) ** 2)
            )

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model, best_model_path)
                print(f"New best model saved with loss {best_loss:.3f} at step {step + 1}")

            if (step + 1) % self.vis_step == 0:
                print(f"Step {step + 1}: Loss {loss.item():.3f}")
                y_hat = self.generate_trajectories(model)
                self.plot_idff_samples_and_trajectories(y_hat, step + 1)

        print(f"Training completed. Best model saved with loss {best_loss:.3f} at {best_model_path}")
        return model

    def evaluate_model(self, model, gamma1_values,gamma2_values, sigma_values, nfes):
        results = []
        for nfe in nfes:
            self.NFE = nfe
            for sigma in sigma_values:
                self.sigma = sigma
                print(f"Evaluating for sigma={sigma}")
                for gamma1 in gamma1_values:
                    for gamma2 in gamma2_values:
                        print(f"Evaluating for gamma1={gamma1}, gamma2={gamma2}")
                        y_hat = self.generate_trajectories(model, gamma1=gamma1, gamma2=gamma2)
                        self.plot_idff_samples_and_trajectories(y_hat, f'sigma_{sigma}_gamma1_{gamma1}_gamma2_{gamma2}')

                        # Calculate metrics
                        metrics = self.compute_metrics(y_hat)
                        true_samples = self.dataset.cpu().numpy().reshape(-1, self.DataDim)
                        generated_samples = y_hat.reshape(-1, self.DataDim)
                        
                        # Compute additional metrics
                        mmd_value = compute_mmd(torch.tensor(true_samples), torch.tensor(generated_samples), sigma=1.0)
                        kl_div_value = compute_kl_divergence(true_samples, generated_samples, bandwidth=1.0)

                        metrics.update({
                            "nfe": nfe,
                            "sigma": sigma,
                            "gamma1": gamma1,
                            "gamma2": gamma2,
                            "MMD": mmd_value,  # Add MMD to metrics
                            "KL-Divergence": kl_div_value  # Add KL-Divergence to metrics
                        })
                        results.append(metrics)

        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(self.save_dir, "evaluation_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Evaluation results saved to {results_csv_path}")


    def generate_trajectories(self, model, gamma1=1, gamma2=10):
        y_hat = np.zeros((self.max_length, self.num_trajectories, self.DataDim))
        
        for t_idx in range(self.max_length):
                sde = SDE_cal(model, input_dim=self.DataDim, gamma1=gamma1, gamma2=gamma2/(1+t_idx), init_ind=t_idx, max_length=self.max_length, dt=0.3, 
                          sigma=self.sigma, device=self.device)

                if t_idx == 0:
                    x1 = torch.randn((self.num_trajectories, self.DataDim)).to(self.device).float()
                else:
                    x1 = torch.tensor(y_hat[t_idx - 1]).to(self.device).float()

                trajectory = torchsde.sdeint(sde, x1, ts=torch.linspace(0, 1, int(self.NFE), device=self.device), dt=sde.dt)
                
                y_hat[t_idx] = trajectory.cpu().detach().numpy()[-1]

        return y_hat
    
    def generate_onestep(self, model,nfe,time_index, gamma1=1, gamma2=10):

        

        sde = SDE_cal(model, input_dim=self.DataDim, gamma1=gamma1, gamma2=gamma2, init_ind=0, max_length=self.max_length, dt=1/nfe, 
                          sigma=self.sigma, device=self.device)

                
        if time_index == 0:
                x1 = torch.randn((self.num_trajectories, self.DataDim)).to(self.device).float()
        else:
                x1 = torch.tensor(self.dataset[time_index - 1]).unsqueeze(0).repeat(self.num_trajectories,1).to(self.device).float()
               
        trajectory = torchsde.sdeint(sde, x1, ts=torch.linspace(0, 1, int(nfe), device=self.device), dt=1/nfe)

        return  trajectory.cpu().detach().numpy()
    def plot_flow_samples(self, prior_sample, color, alpha, marker, ax, title=""):
        z = prior_sample.cpu()
        ax.scatter(z[:, 0], z[:, 1], c=color, s=1, marker=marker, alpha=alpha)
        ax.set_title(title)
        ax.invert_yaxis()

    def plot_idff_samples_and_trajectories(self, y_hat, step):
        rgn_thr = self.rgn_thr

        # First set of plots: Generated samples vs Dataset
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        self.plot_flow_samples(torch.tensor(y_hat.reshape([-1, 2])), 'g', 0.2, '.', axes[0], title="Generated Samples")
        self.plot_flow_samples(self.dataset.reshape([-1, 2]).cpu(), 'k', 0.2, '.', axes[1], title="Data")

        axes[0].set_xlim([-180, 180])
        axes[0].set_ylim([-180, 180])

        plot_path_base = os.path.join(self.save_dir, f"generated_step_{step}")
        plt.savefig(f"{plot_path_base}.png")
        plt.savefig(f"{plot_path_base}.svg", format='svg')
        plt.close()

        # Second set of plots: Trajectories
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True, sharey=False)
        indx_angle = 10

        for kk in range(self.num_trajectories):
            axes[0].plot(y_hat[:, kk, :].reshape(self.max_length, int(self.DataDim // 2), 2)[:, indx_angle, 0], color='r', alpha=0.9)
            axes[1].plot(y_hat[:, kk, :].reshape(self.max_length, int(self.DataDim // 2), 2)[:, indx_angle, 1], color='r', alpha=0.9)

        dataset_np = self.dataset.cpu().numpy().reshape(self.max_length, int(self.DataDim // 2), 2)
        axes[0].plot(dataset_np[:, indx_angle, 0], color='g', alpha=0.9)
        axes[0].axvline(x=rgn_thr, color='k', linestyle='--')
        axes[0].axvline(x=y_hat.shape[0] - rgn_thr, color='b', linestyle='--')

        axes[1].plot(dataset_np[:, indx_angle, 1], color='g', alpha=0.9)
        axes[1].axvline(x=y_hat.shape[0] - rgn_thr, color='b', linestyle='--')
        axes[1].axvline(x=rgn_thr, color='k', linestyle='--')

        plt.savefig(f"{plot_path_base}_trj.png")
        plt.savefig(f"{plot_path_base}_trj.svg", format='svg')
        plt.show()

    def compute_metrics(self, y_hat):
        all_cc, all_mse, all_mae = [], [], []
        dataset_np = self.dataset.cpu().numpy()

        for ii in range(y_hat.shape[1]):
            for jj in range(y_hat.shape[2]):
                cc = np.corrcoef(y_hat[:, ii, jj], dataset_np[:, jj])[0, 1]
                all_cc.append(cc)
                mse = np.sqrt(np.nanmean((y_hat[:, ii, jj] - dataset_np[:, jj]) ** 2))
                all_mse.append(mse)
                mae = np.nanmean(np.abs(y_hat[:, ii, jj] - dataset_np[:, jj]))
                all_mae.append(mae)

        return {
            "CC_mean": np.mean(all_cc),
            "CC_std": np.std(all_cc),
            "RMSE_mean": np.mean(all_mse),
            "RMSE_std": np.std(all_mse),
            "MAE_mean": np.mean(all_mae),
            "MAE_std": np.std(all_mae),
        }
    def plot_overlayed_trajectories(self, model, gamma1, gamma2, step_label="comparison"):
        """
        Plot overlayed trajectories for different gamma settings:
        (1) Both gamma1 and gamma2 active
        (2) Only gamma1 active
        (3) Both gamma1 and gamma2 zero
        """
        y_hat_full = self.generate_trajectories(model, gamma1=gamma1, gamma2=gamma2)
        y_hat_gamma1_only = self.generate_trajectories(model, gamma1=gamma1, gamma2=0)
        y_hat_no_gamma = self.generate_trajectories(model, gamma1=0, gamma2=0)

        dataset_np = self.dataset.cpu().numpy().reshape(self.max_length, int(self.DataDim // 2), 2)

        rgn_thr = self.rgn_thr
        indx_angle = 10  # Choose an example angle to plot

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Plot for first component
        for kk in range(self.num_trajectories):
            axes[0].plot(y_hat_full[:, kk, :].reshape(nfe, int(self.DataDim // 2), 2)[:, indx_angle, 0],
                        color='r', label='Gamma1 & Gamma2' if kk == 0 else "", alpha=0.6)
            axes[0].plot(y_hat_gamma1_only[:, kk, :].reshape(self.max_length, int(self.DataDim // 2), 2)[:, indx_angle, 0],
                        color='b', label='Only Gamma1' if kk == 0 else "", alpha=0.6)
            axes[0].plot(y_hat_no_gamma[:, kk, :].reshape(self.max_length, int(self.DataDim // 2), 2)[:, indx_angle, 0],
                        color='g', label='No Gamma' if kk == 0 else "", alpha=0.6)

        # Plot the true dataset
        axes[0].plot(dataset_np[:, indx_angle, 0], color='k', label='True', linewidth=2)

        axes[0].axvline(x=rgn_thr, color='k', linestyle='--')
        axes[0].axvline(x=self.max_length - rgn_thr, color='b', linestyle='--')
        axes[0].set_ylabel("Component 1")
        axes[0].legend()

        # Plot for second component
        for kk in range(self.num_trajectories):
            axes[1].plot(y_hat_full[:, kk, :].reshape(self.max_length, int(self.DataDim // 2), 2)[:, indx_angle, 1],
                        color='r', label='Gamma1 & Gamma2' if kk == 0 else "", alpha=0.6)
            axes[1].plot(y_hat_gamma1_only[:, kk, :].reshape(self.max_length, int(self.DataDim // 2), 2)[:, indx_angle, 1],
                        color='b', label='Only Gamma1' if kk == 0 else "", alpha=0.6)
            axes[1].plot(y_hat_no_gamma[:, kk, :].reshape(self.max_length, int(self.DataDim // 2), 2)[:, indx_angle, 1],
                        color='g', label='No Gamma' if kk == 0 else "", alpha=0.6)

        axes[1].plot(dataset_np[:, indx_angle, 1], color='k', label='True', linewidth=2)

        axes[1].axvline(x=rgn_thr, color='k', linestyle='--')
        axes[1].axvline(x=self.max_length - rgn_thr, color='b', linestyle='--')
        axes[1].set_ylabel("Component 2")
        axes[1].set_xlabel("Time Step")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"overlayed_trajectories_{step_label}.png")
        plt.savefig(save_path)
        plt.savefig(save_path.replace('.png', '.svg'))
        plt.show()
        print(f"Overlayed trajectories saved at {save_path}")
    

    def plot_phase_space_overlayed_trajectories(self, model, nfe,trajs, time_indx, gamma1, gamma2, step_label="phase_plot"):
        """
        Plot phase space trajectories (Component 1 vs Component 2)
        using chained arrows following the trajectory points.
        Only completed trajectories are plotted. Start and end points are highlighted.
        Axis limits are automatically set to fit all trajectories.
        """

        # Generate trajectories
        y_hat_full = self.generate_onestep(model, nfe, time_indx, gamma1=gamma1, gamma2=gamma2)
        y_hat_gamma1_only = self.generate_onestep(model, nfe, time_indx, gamma1=gamma1, gamma2=0)
        y_hat_no_gamma = self.generate_onestep(model, nfe, time_indx, gamma1=0, gamma2=0)

        # True dataset
        dataset_np = self.dataset.cpu().numpy().reshape(self.max_length, int(self.DataDim // 2), 2)

        indx_angle = 10  # Choose one angle (trajectory component)

        fig, ax = plt.subplots(figsize=(8, 6))

        all_points = []  # To collect all points for setting axis limits

        # Helper function to plot chain of arrows following trajectory
        def plot_trajectory_arrows(traj, color, label):
            traj = traj.reshape(nfe, int(self.DataDim // 2), 2)[:, indx_angle, :]  # shape (nfe, 2)

            # Check if the trajectory is valid (no NaNs)
            if np.isnan(traj).any():
                return  # skip incomplete trajectories

            all_points.extend(traj)

            for i in range(len(traj) - 1):
                start = traj[i]
                end = traj[i + 1]
                dx = end[0] - start[0]
                dy = end[1] - start[1]

                ax.quiver(start[0], start[1], dx, dy,
                        angles='xy', scale_units='xy', scale=1,
                        color=color, alpha=0.8, width=0.003,
                        label=label if i == 0 else None)

            # Highlight the start point
            ax.scatter(traj[0, 0], traj[0, 1], color=color, edgecolor='black', s=60, marker='o')

            # Highlight the end point
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, edgecolor='black', s=80, marker='X')

        # Plot arrows for each trajectory type
        for kk in trajs:
            plot_trajectory_arrows(y_hat_full[:, kk, :], 'red', 'Gamma1 & Gamma2' if kk == 0 else None)
            plot_trajectory_arrows(y_hat_gamma1_only[:, kk, :], 'blue', 'Only Gamma1' if kk == 0 else None)
            plot_trajectory_arrows(y_hat_no_gamma[:, kk, :], 'green', 'No Gamma' if kk == 0 else None)

        # Also include the true dataset point
        true_point = dataset_np[time_indx, indx_angle, :]
        all_points.append(true_point)

        ax.scatter(true_point[0], true_point[1],
                color='black', s=150, marker='*', label='True Point', edgecolor='k')

        ax.set_xlabel("Component 1 (x)")
        ax.set_ylabel("Component 2 (y)")
        ax.set_title("Phase Space Trajectories with Chained Arrows")

        # Set axis limits dynamically
        all_points = np.array(all_points)
        xmin, ymin = np.min(all_points, axis=0)
        xmax, ymax = np.max(all_points, axis=0)

        # Make it square (same range for x and y)
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        max_range = max(xmax - xmin, ymax - ymin) / 2 * 1.1  # add 10% padding

        ax.set_xlim([x_center - max_range, x_center + max_range])
        ax.set_ylim([y_center - max_range, y_center + max_range])

        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')  # Keep the aspect ratio square

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"phase_plot_overlayed_{step_label}.png")
        plt.savefig(save_path)
        plt.savefig(save_path.replace('.png', '.svg'))
        plt.show()

        print(f"Phase plot saved at {save_path}")

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model, input_dim, sigma=1, init_ind=1, max_length=1, dt=0.1, gamma1=1, gamma2=1, device='cpu'):
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.init_ind = torch.tensor([init_ind])
        self.max_length = max_length
        self.dt = dt
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.device = device
        self.input_dim = input_dim

    def f(self, t, y):
        y = y.view(-1, self.input_dim)
        t_tensor = (t * torch.ones((y.shape[0], 1))).to(self.device)
        init_tensor = (self.init_ind * torch.ones((y.shape[0], 1))).to(self.device)
        sigma_t2 = (self.sigma ** 2) * t * (1 - t)
        predictions = self.model(torch.cat([y, t_tensor], dim=-1), init_tensor)

        if t == 0:
            self.x0=y
            return predictions - y
        elif t == 1:
            return (predictions - y) / self.dt
        else:
            return torch.sqrt(1 - sigma_t2) * (predictions - y) / ((1 - t)) - 0.5 * (2 * self.gamma1 - 1) * (self.x0)*sigma_t2/(self.sigma ** 2) - self.gamma2*sigma_t2/(self.sigma ** 2)

    def g(self, t, y):
        return torch.ones_like(y)*180 * (self.sigma ** 2) * torch.sqrt(t * (1 - t))

class SDE_cal(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model, input_dim, sigma=1, init_ind=1, max_length=1, dt=0.1, gamma1=1, gamma2=1, device='cpu'):
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.init_ind = torch.tensor([init_ind], device=device)
        self.max_length = max_length
        self.dt = dt
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.device = device
        self.input_dim = input_dim

    def f(self, t, y):
        y = y.view(-1, self.input_dim).to(self.device)
        y.requires_grad_(True)  # Enable gradient tracking
        t_tensor = (t * torch.ones((y.shape[0], 1), device=self.device))
        init_tensor = (self.init_ind * torch.ones((y.shape[0], 1), device=self.device))
        sigma_t2 = (self.sigma ** 2) * t * (1 - t)

        predictions = self.model(torch.cat([y, t_tensor], dim=-1), init_tensor)

        if t == 0:

            return predictions - y  # Replace self.x0 with first-order gradient
        elif t == 1:
            return (predictions - y) / self.dt
        else:
            # Compute first-order gradient
            first_order_grad = torch.autograd.grad(
                predictions, y, grad_outputs=torch.ones_like(predictions), create_graph=True
            )[0]

            # Compute second-order gradient
            second_order_grad = torch.autograd.grad(
                first_order_grad, y, grad_outputs=torch.ones_like(first_order_grad), create_graph=True
            )[0]

            return (
                torch.sqrt(1 - sigma_t2) * (predictions - y) / ((1 - t))
                - 0.5 * (2 * self.gamma1 - 1) * first_order_grad * sigma_t2 / (self.sigma ** 2)  # First-order gradient term
                - self.gamma2 * second_order_grad * sigma_t2 / (self.sigma ** 2)  # Second-order gradient term
            )

    def g(self, t, y):
        return torch.ones_like(y) * 180 * (self.sigma ** 2) * torch.sqrt(t * (1 - t))

# Usage
if __name__ == "__main__":
    dataset_names = ["polyALA"]#, "villain_dynamics", "beta_hairpin_dynamics"]  # Example dataset names
    sigma_values = [0.5]
    gamma1_values = [  0.2,]
    gamma2_values= [ 5]
    nfes=[3,]

    for dataset_name in dataset_names:
        trajectory_generator = TrajectoryGenerator(
            dataset_name=dataset_name,
            sigma=0.5,
            nfe=3,
            batch_size=10,
            num_trajectories=5,
            save_dir="MD/",
            model_name=f"MD_trained_{dataset_name}",
            use_cuda=False
        )

        model_path = os.path.join(trajectory_generator.save_dir, f"{trajectory_generator.model_name}.pt")

        if not os.path.exists(model_path):
            model = trajectory_generator.train_model()
        else:
            model = torch.load(model_path)

        trajectory_generator.evaluate_model(model, gamma1_values,gamma2_values, sigma_values,nfes)
#         trajectory_generator.plot_overlayed_trajectories(
#     model,
#     gamma1=gamma1_values[0],
#     gamma2=gamma2_values[0],
#     step_label="final"
# )
        trajectory_generator.plot_phase_space_overlayed_trajectories(
    model,nfe=3,time_indx=10,trajs=[0],
    gamma1=gamma1_values[0],
    gamma2=gamma2_values[0],
    step_label="initial"
)
