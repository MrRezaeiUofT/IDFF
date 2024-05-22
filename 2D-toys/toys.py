import math
import os
import time

from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
import  torchsde

use_cuda=True
device = torch.device("cuda" if use_cuda else "cpu")
class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model,sigma=1,model_name='non', device='cpu'):
        super().__init__()
        self.model = model
        self.device=device
        self.model_name=model_name
        self.sigma = sigma



    # Drift
    def f(self, t, y):
        y = y.view(-1, 2)


        predicts = self.model(torch.cat([y,y, t*torch.ones((y.shape[0],1)).to(self.device)], dim=-1))
        if t == 0:

            self.x0 = y
            outs = predicts[:, :2]- self.x0
        elif t ==1:
            outs = (predicts[:, :2] - y) / 0.1
        else:

            outs = (1-(self.sigma**2)*t*(1-t))*(predicts[:, :2] - y)/(1-t) + (predicts[:, 2:])* (self.sigma **2)*torch.sqrt((t)*(1-t))


        return outs.flatten(start_dim=1)

    # Diffusion
    def g(self, t, y):

        return torch.ones_like(y) *(self.sigma **2)*torch.sqrt((t)*(1-t))
savedir = "8gaussians/"
os.makedirs(savedir, exist_ok=True)

sigma = 0.2 # sigma_0
dim = 4 # 2 for x_hat and 2 for sigma models
batch_size = 256
model = MLP(dim=dim, time_varying=True).to(device)
optimizer = torch.optim.Adam(model.parameters())
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

start = time.time()
for k in range(40000):
    optimizer.zero_grad()



    x1 = sample_8gaussians(batch_size).to(device)
    x0 = torch.randn_like(x1).to(device)
    t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
    lambda_t = FM.compute_lambda(t)
    sigma_t = FM.compute_sigma_t(t)[:, None,]

    outs = model(torch.cat([xt,xt, t[:, None]], dim=-1))
    vt = outs[:, :2]
    st = outs[:, 2:]
    flow_loss = torch.mean((vt - FM.x1) ** 2)
    score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
    loss = 5 * flow_loss + score_loss
    loss.backward()


    optimizer.step()

    if (k + 1) % 2000 == 0:

        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end

        sde = SDE(model,model_name="IDFF", sigma=1*sigma,device=device)
        with torch.no_grad():
            x1 = torch.randn((4098,2)).to(device)
            traj = torchsde.sdeint(
                sde,
                # x0.view(x0.size(0), -1),
                x1,
                ts=torch.linspace(0, 1, 10, device=device),
                dt=.1,
            )
            plot_trajectories(traj.cpu().numpy(),x1.cpu().numpy())
            plt.savefig( savedir + f"_generated_IDFF_images_step_{k}.png")
            plt.savefig(savedir + f"_generated_IDFF_images_step_{k}.svg",  format='svg')
            plt.close()
torch.save(model, f"{savedir}/IDFF.pt")