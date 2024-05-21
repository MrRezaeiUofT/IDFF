
import math
import os
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
import  torchsde
from sklearn.neighbors import KernelDensity
import numpy as np
def plt_flow_samples(prior_sample,color,alpha,marker, ax, npts=100, memory=100, kde_enable=True, title="", device="cpu"):
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
        kde = KernelDensity(bandwidth=.1, kernel='gaussian')
        kde.fit(zk)

        # Compute the log density values for the grid points
        log_density = kde.score_samples(grid_points)
        density = log_density.reshape(xx.shape)
        ax.imshow(density.T, cmap=color,  # ,extent=(-2, 3, -2, 3),
                  # interpolation='nearest',
                  origin='lower')
    else:

        ax.scatter(zk[:, 0], zk[:, 1], c=color, s=1,marker=marker, alpha=alpha)
        # ax.axis('off')
    ax.invert_yaxis()
    # ax.get_xaxis().set_ticks(['p)

use_cuda=True
train_enable=False
savedir = "MD/"
device = torch.device("cuda" if use_cuda else "cpu")
dataset_name="polyALA"
sigma = 0.1

batch_size = 10
trjs_s=2
scale=torch.tensor([.1]).to(device)

with open(savedir+'/'+dataset_name+"_dataset.p", 'rb') as file:
    dataset = pickle.load(file)
orig_trajs=dataset['angles'][1].reshape([dataset['sim_length'],-1])
dataset = torch.tensor(orig_trajs).squeeze().float().to(device)
DataDim=orig_trajs.shape[-1]
dim = 2*DataDim
class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model,sigma=1,init_ind=1,max_length=1,dt=.1,model_name='non', device='cpu'):
        super().__init__()
        self.model = model
        self.device=device
        self.model_name=model_name
        self.sigma = sigma
        self.max_length=max_length
        self.init_ind=torch.tensor([init_ind])
        self.dt=dt



    # Drift
    def f(self, t, y):
        y = y.view(-1, DataDim)
        predicts = self.model(torch.cat([y, y, (t) * torch.ones((y.shape[0], 1)).to(self.device)], dim=-1),
                              (self.init_ind * torch.ones((y.shape[0], 1))).to(self.device))
        if t == 0:


            self.x0 = y

            outs = (predicts[:, :DataDim]-y)
        elif ((t ==  1)):


            outs = (predicts[:, :DataDim] - y) / self.dt
        else:

            outs = (1 - (self.sigma ** 2) * t * (1 - t)) * (predicts[:, :DataDim] - y) / (1 - t) \
                   - (predicts[:, DataDim:]/2) *  (self.sigma **2) * torch.sqrt((t) * (1 - t))


        return outs.flatten(start_dim=1)

    # Diffusion
    def g(self, t, y):

        return torch.ones_like(y) *(self.sigma **2)*torch.sqrt((t)*(1-t))

os.makedirs(savedir, exist_ok=True)



FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)


# Solve the differential equations

start = time.time()


max_length=dataset.shape[0]
model = MLP_Embedding(dim=dim,w=128,time_embed=max_length, time_varying=True).to(device)
optimizer = torch.optim.Adam(model.parameters())
if train_enable:
    for k in range(20000):
        optimizer.zero_grad()


        index_sel=torch.randint(1, max_length, (batch_size,)).to(device)

        x0 = dataset[index_sel-1].to(device)
        # x0[index_sel==1]=scale*torch.randn_like(x0[index_sel==1]).to(device)
        x1 = dataset[index_sel].to(device)
        # t = torch.rand(x0.shape[0]).type_as(x0)
        # xt = FM.sample_xt(x0, x1, t, x0)
        t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
        lambda_t = FM.compute_lambda(t)
        sigma_t = FM.compute_sigma_t(t)[:, None,]

        outs = model(torch.cat([xt,xt, t[:, None]], dim=-1),index_sel)
        vt = outs[:, :DataDim]
        st = outs[:, DataDim:]
        t_w=1/(1-t[:, None]+1e-2)
        numbers = torch.tensor([0, 2, -2]).to(device)

        # Choose a random index
        index_shif = torch.randint(0, len(numbers), (xt.shape[0],)).to(device)
        shifts=torch.ones_like(x1)*index_shif[:,None]*torch.pi
        flow_loss = torch.mean(((vt - x1) ** 2))
        score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
        loss = 2 * flow_loss + score_loss
        loss.backward()


        optimizer.step()

        if (k+1 ) % 10000 == 0:

            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end


            y_hat=np.zeros((max_length,trjs_s,DataDim))
            with torch.no_grad():
                for tt in range(max_length):
                    sde = SDE(model, model_name="IDFF", init_ind=tt,max_length=max_length,dt=0.3, sigma=0 * sigma, device=device)
                    if tt==0:
                        x1 = torch.randn((trjs_s,y_hat.shape[-1])).to(device).float()
                    else:
                        x1 = torch.tensor(y_hat[tt-1]).to(device).float()

                    traj = torchsde.sdeint(
                        sde,
                        x1,
                        ts=torch.linspace(0, 1, 3, device=device),
                        dt=sde.dt,
                    )
                    y_hat[tt]= traj.cpu().numpy()[-1]

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            plt_flow_samples(torch.tensor(y_hat.reshape([-1, 2])), axes[0], npts=50, memory=50000, kde_enable=False,
                             title="generated_samples_by_alternator", device="cpu")
            plt_flow_samples(dataset.reshape([-1, 2]).cpu(), axes[1], npts=50, memory=50000, kde_enable=False, title="data",
                             device="cpu")
            # plt.tight_layout()
            plt.savefig(savedir + f"_generated_IDFF_images_step_{k}.png")
            plt.savefig(savedir + f"_generated_IDFF_images_step_{k}.svg", format='svg')

    torch.save(model, f"{savedir}/IDFF_MD_v1.pt")


#################test
else:
    model=(torch.load(f"{savedir}/IDFF_MD_v1.pt", map_location=torch.device('cpu'))).to(device)
    y_hat=np.zeros((max_length,trjs_s,DataDim))
    with torch.no_grad():
            for tt in range(max_length):
                sde = SDE(model, model_name="bb", init_ind=tt,max_length=max_length,dt=.3, sigma=1 * sigma, device=device)
                if tt==0:
                    x1 = torch.randn((trjs_s,y_hat.shape[-1])).to(device).float()
                else:
                    x1 = torch.tensor(y_hat[tt-1]).to(device).float()

                traj = torchsde.sdeint(
                    sde,
                    # x0.view(x0.size(0), -1),
                    x1,
                    ts=torch.linspace(0, 1, 2, device=device),
                    dt=sde.dt,
                )
                y_hat[tt]= traj.cpu().numpy()[-1]
    rgn_thr=30
    fig, axes = plt.subplots(1, 2, figsize=(8, 4),sharex=True,sharey=True)
    plt_flow_samples(torch.tensor(y_hat.reshape([-1, 2])),'g',.1,'.', axes[0], npts=50, memory=50000, kde_enable=False,
                         title="generated_samples_by_alternator", device="cpu")
    plt_flow_samples(dataset.reshape([-1, 2]).cpu(),'g',.2,'.', axes[1], npts=50, memory=50000, kde_enable=False, title="data",
                         device="cpu")

    plt_flow_samples(torch.tensor(y_hat[:rgn_thr].reshape([-1, 2])), 'black', .4, 'o', axes[0], npts=50, memory=50000,
                     kde_enable=False,
                     title="generated_samples_by_alternator", device="cpu")
    plt_flow_samples(dataset[:rgn_thr].reshape([-1, 2]).cpu(), 'black', .8, 'o', axes[1], npts=50, memory=50000,
                     kde_enable=False, title="data",
                     device="cpu")

    plt_flow_samples(torch.tensor(y_hat[-rgn_thr:].reshape([-1, 2])), 'blue', .4, '*', axes[0], npts=50, memory=50000,
                     kde_enable=False,
                     title="generated_samples_by_alternator", device="cpu")
    plt_flow_samples(dataset[-rgn_thr:].reshape([-1, 2]).cpu(), 'blue', .8, '*', axes[1], npts=50, memory=50000,
                     kde_enable=False, title="data",
                     device="cpu")
        # plt.tight_layout()
    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-180, 180])
    plt.savefig(savedir + f"_generated_IDFF_images_step_test.png")
    plt.savefig(savedir + f"_generated_IDFF_images_step_test.svg", format='svg')
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True, sharey=False)
    indx_angle=10
    for kk in range(trjs_s):
        axes[0].plot(y_hat[:, kk, :].reshape(max_length,int(DataDim//2),2)[:, indx_angle,0], color='r', alpha=0.9)
        axes[1].plot(y_hat[:, kk, :].reshape(max_length, int(DataDim // 2), 2)[:, indx_angle, 1], color='r', alpha=0.9)
    axes[0].plot(dataset.cpu().numpy().reshape(max_length,int(DataDim//2),2)[:, indx_angle,0], color='g', alpha=0.9)
    axes[0].axvline(x=rgn_thr, color='k', linestyle='--')
    axes[0].axvline(x=y_hat.shape[0] - rgn_thr, color='b', linestyle='--')
    axes[1].plot(dataset.cpu().numpy().reshape(max_length, int(DataDim // 2), 2)[:, indx_angle, 1], color='g',
                 alpha=0.9)
    axes[1].axvline(x=y_hat.shape[0]-rgn_thr, color='b', linestyle='--')
    axes[1].axvline(x=rgn_thr, color='k', linestyle='--')
    # axes[1].set_xlim(-180,180)
    # axes[1].set_ylim(-180, 180)
    plt.savefig(savedir + f"_generated_IDFF_images_step_test_trj.png")
    plt.savefig(savedir + f"_generated_IDFF_images_step_test_trj.svg", format='svg')


    all_cc=[]
    all_mse=[]
    all_mae=[]
    for ii in range(y_hat.shape[1]):
        # Calculate cross-correlation (CC)
        for jj in range(y_hat.shape[2]):
            cc = np.corrcoef(y_hat[:,ii,jj], dataset.detach().cpu().numpy()[:,jj])[0, 1]
            all_cc.append(cc)

            # Calculate mean squared error (MSE)
            mse =np.sqrt( np.nanmean((y_hat[:,ii,jj] - dataset.detach().cpu().numpy()[:,jj]) ** 2))
            all_mse.append(mse)
            # Calculate mean absolute error (MAE)
            mae = np.nanmean(np.abs(y_hat[:,ii,jj] - dataset.detach().cpu().numpy()[:,jj]))
            all_mae.append(mae)


    print("Cross-correlation (CC)=%f+-%f:"%( np.array(all_cc).mean(),np.array(all_cc).std()))
    print("Mean Squared Error (RMSE)=%f+-%f:"%( np.array(all_mse).mean(),np.array(all_mse).std()))
    print("Mean Absolute Error (MAE)=%f+-%f:"%( np.array(all_mae).mean(),np.array(all_mae).std()))

