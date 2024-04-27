import math
import os
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
import  torchsde
from spiral2d import generate_spiral2d
from scipy.integrate import solve_ivp
DataDim=2
use_cuda=True
device = torch.device("cuda" if use_cuda else "cpu")
class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model,sigma=1,init_ind=1,max_length=1,model_name='non', device='cpu'):
        super().__init__()
        self.model = model
        self.device=device
        self.model_name=model_name
        self.sigma = sigma
        self.max_length=max_length
        self.init_ind=torch.tensor([init_ind])



    # Drift
    def f(self, t, y):
        y = y.view(-1, DataDim)
        predicts = self.model(torch.cat([y, y, (t) * torch.ones((y.shape[0], 1)).to(self.device)], dim=-1),
                              (self.init_ind * torch.ones((y.shape[0], 1))).to(self.device))
        if t == 0:


            self.x0 = y

            outs = predicts[:, :DataDim]- self.x0- (predicts[:, DataDim:]/2)*1*torch.sqrt((t)*(1-t))
        elif ((t ==  1)):



            outs = (predicts[:, :DataDim] - y)/.01  - (predicts[:, DataDim:]/2)*1*torch.sqrt((t)*(1-t))
        else:




            outs = (predicts[:, :DataDim] - y)/(1-t) - (predicts[:, DataDim:]/2)*1*torch.sqrt((t)*(1-t))


        return outs.flatten(start_dim=1)

    # Diffusion
    def g(self, t, y):

        return torch.ones_like(y) *self.sigma*1*torch.sqrt((t)*(1-t))
savedir = "spiral/"
os.makedirs(savedir, exist_ok=True)

sigma = 0.1
dim = 2*DataDim
batch_size = 10
trjs_s=10

FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

scale=torch.tensor([.1]).to(device)
# Solve the differential equations
scaler = MinMaxScaler(feature_range=(-1, 1))

start = time.time()
orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=1,
        start=4*np.random.uniform(size=1) * np.pi,
        stop=6 * np.pi,
        noise_std=.3,
        a=0, b=.3
    )
dataset = torch.tensor(orig_trajs).squeeze().float().to(device)
max_length=dataset.shape[0]
model = MLP_Embedding(dim=dim,w=32,time_embed=max_length, time_varying=True).to(device)
optimizer = torch.optim.Adam(model.parameters())
for k in range(15000):
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
    flow_loss = torch.mean(((vt - x1) ** 2))
    score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
    loss = 2 * flow_loss + score_loss
    loss.backward()


    optimizer.step()

    if (k+1 ) % 1000 == 0:

        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end


        y_hat=np.zeros((max_length,trjs_s,DataDim))
        with torch.no_grad():
            for tt in range(max_length):
                sde = SDE(model, model_name="bb", init_ind=tt,max_length=max_length, sigma=0 * sigma, device=device)
                if tt==0:
                    x1 = torch.randn((trjs_s,y_hat.shape[-1])).to(device).float()
                else:
                    x1 = torch.tensor(y_hat[tt-1]).to(device).float()+1*scale*torch.randn((trjs_s,y_hat.shape[-1])).to(device).float()

                traj = torchsde.sdeint(
                    sde,
                    # x0.view(x0.size(0), -1),
                    x1,
                    ts=torch.linspace(0, 1, 3, device=device),
                    dt=.33,
                )
                y_hat[tt]= traj.cpu().numpy()[-1]

        fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
        for kk in range(trjs_s):
            ax[0].plot(y_hat[1:,kk,0], y_hat[1:,kk,1], color='r', alpha=0.9)
        ax[1].plot(dataset.cpu().numpy()[:, 0],
                dataset.cpu().numpy()[:, 1],
                #
                color='g', alpha=0.9)
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_title('Spiral time series ')
        plt.savefig( savedir + f"_generated_FM_images_step_{k}.png")
        plt.savefig(savedir + f"_generated_FM_images_step_{k}.svg", format='svg')

        # fig, ax= plt.subplots(3,1)
        #
        # for kk in range(batch_size):
        #     ax[0].plot(y_hat[:, kk, 0], color='b', alpha=0.4)
        #     ax[1].plot(y_hat[:, kk, 1], color='b', alpha=0.4)

        # ax[0].plot(dataset.cpu().numpy()[:, 0], alpha=0.7)
        # ax[1].plot(dataset.cpu().numpy()[:, 1], alpha=0.7)



        # plt.savefig(savedir + f"sep_generated_FM_images_step_{k}.png")
        # plt.close()
torch.save(model, f"{savedir}/bb_v1.pt")