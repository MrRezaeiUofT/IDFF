import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
import pickle

def generate_spiral2d(nspiral=10,
                      ntotal=100,
                      nsample=1,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.
    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check
    Returns:
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts



if __name__ == "__main__":

    nspiral = 10
    start = 4 * np.pi
    stop = 6 * np.pi
    noise_std = .3
    a = 0.
    b = .3
    ntotal = 10
    nsample = 10
    device = torch.device('cuda:'
                          if torch.cuda.is_available() else 'cpu')
    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=nspiral,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a=a, b=b
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)
    number_of_obsr = 30
    trj_noise_std = .05

    selected_trj = orig_trajs[4]
    selected_trj=selected_trj-selected_trj[0,:]
    noisy_trj = np.zeros_like(selected_trj)
    obsr_cov = torch.linspace(.19, .2, number_of_obsr) * torch.eye(number_of_obsr)
    mvn = MultivariateNormal(torch.zeros(number_of_obsr),
                             obsr_cov)
    observations = np.zeros((noisy_trj.shape[0], noisy_trj.shape[1], number_of_obsr))

    for state_dim in range(selected_trj.shape[1]):
        noisy_trj[:,state_dim] = selected_trj[:,state_dim] + trj_noise_std* np.random.randn(selected_trj.shape[0])
        observations[:,state_dim,:] = (noisy_trj[:,state_dim].reshape([-1,1]) +
        np.linspace(.8,1,noisy_trj.shape[0]).reshape([-1,1])*mvn.sample((noisy_trj.shape[0],)).detach().numpy())

    plt.figure()

    for ii in range(number_of_obsr):
        plt.scatter(observations[:, 0,:], observations[:, 1,:],marker='o', c='k',s=3, alpha=.1)
    plt.plot(selected_trj[:, 0], selected_trj[:, 1],
             'g', label='true trajectory')
    plt.scatter(noisy_trj[:, 0], noisy_trj[:, 1], label='noisy state', marker='*', c='r', s=20)

    plt.legend()
    plt.savefig('vis.png', dpi=500)
    print('Saved visualization figure at {}'.format('vis.png'))
    plt.show()
    spiral2d_dataset= {'state': noisy_trj,
             'observation': observations,
             'state_dim':2,
             'number_of_obsr':number_of_obsr,
             'state_noise_std':trj_noise_std,
             'observation_cov':obsr_cov}

    pickle.dump(spiral2d_dataset, open("spiral2d_dataset.p", "wb"))
    spiral2d_dataset_v2 = pickle.load(open("spiral2d_dataset.p", "rb"))