import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdyn
from torchdyn.datasets import generate_moons
rng = np.random.RandomState()
# Implement some helper functions


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def pinwheel(n):
    radial_std = 0.3
    tangential_std = 0.1
    num_classes = 5
    num_per_class = n // 5
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = rng.randn(num_classes * num_per_class, 2) \
               * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return torch.tensor(2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations)))

def spirals_2d(n):
    n = np.sqrt(np.random.rand(n // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n // 2, 1) * 0.5
    d1y = np.sin(n) * n + np.random.rand(n // 2, 1) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.1
    return torch.tensor(x)

def checkerboard(n):
    x1 = np.random.rand(n) * 4 - 2
    x2_ = np.random.rand(n) - np.random.randint(0, 2, n) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return torch.tensor(np.concatenate([x1[:, None], x2[:, None]], 1) * 2)

def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()

def sample_pinwheel(n):
    return pinwheel(n ).float()

def sample_spirals(n):
    return spirals_2d(n ).float()

def sample_checkerboard(n):
    return checkerboard(n ).float()

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))



def plot_trajectories(traj,true):
    """Plot trajectories of some selected samples."""
    n = traj.shape[1]
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.2, c="g")
    x=traj[:, :n, 0]
    y=traj[:, :n, 1]
    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color='b',width=0.01, headwidth=5)
    # plt.scatter(traj[:-1, :n, 0], traj[:-1, :n, 1], s=0.2, alpha=0.5, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="black")
    # plt.scatter(true[:, 0], true[:, 1], s=4, alpha=1, c="g")
    # plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xlim(-5,5)
    plt.ylim(-5, 5)
    plt.xticks([])
    plt.yticks([])
def plot_trajectories_m(traj, true, moment_traj, finals):
    """Plot trajectories of some selected samples with arrows representing moment_traj."""
    n = traj.shape[1]

    plt.scatter(finals[:, 0], finals[:, 1], s=.8, alpha=.3, c="black")
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=1, c="g")
    x = traj[:, :n, 0]
    y = traj[:, :n, 1]
    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color='b',width=0.01, headwidth=2, alpha=.5)
    for i in range(n):
        for j in range(traj.shape[0]):
            plt.arrow(x[j, i], y[j, i], moment_traj[j, i, 0],moment_traj[j, i, 1], color='r', alpha=0.8, width=0.01, head_width=0.2)

    # plt.scatter(true[:, 0], true[:, 1], s=4, alpha=.1, c="g")
    # plt.legend(["Prior samples", "IDFF", "IDFF momentum term"])
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    plt.xticks([])
    plt.yticks([])



def plt_flow_samples(prior_sample, npts=100, memory=100, kde_enable=True, title="", device="cpu"):
    z = prior_sample.to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory ** 2)):
        zk.append(z[ii])
    zk = torch.cat(zk, 0).cpu().numpy()
    x_min, x_max = prior_sample[:, 0].min(), prior_sample[:, 0].max()
    y_min, y_max = prior_sample[:, 1].min(), prior_sample[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, npts),
                         np.linspace(y_min, y_max, npts))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)

    if kde_enable:
        # Fit a kernel density estimator to the data
        kde = KernelDensity(bandwidth=0.2)
        kde.fit(zk)

        # Compute the log density values for the grid points
        log_density = kde.score_samples(grid_points)

        # Reshape the log density values to match the grid shape
        density = np.exp(log_density)
        density = density.reshape(xx.shape)
        plt.imshow(density.T,  # ,extent=(-2, 3, -2, 3),
                  # interpolation='nearest',
                  origin='lower')
    else:
        # hist, x_edges, y_edges = np.histogram2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=(npts, npts))
        # ax.imshow(hist, cmap='copper',
        #           interpolation='nearest',
        #           origin='lower')
        # copper_color = (0.5, 0.3, 0.1)
        burnt_orange = "#cc5500"
        plt.scatter(zk[:, 0], zk[:, 1], c='k', s=.05, alpha=.02)
