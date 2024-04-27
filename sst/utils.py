import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
import copy
import xarray as xr
import xskillscore
import torch
from torchdyn.core import NeuralODE
from torchvision.transforms import ToPILImage
import torchsde
# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m


def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z


def create_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def random_uniform_like(tensor, min_val, max_val):
    return (max_val - min_val) * torch.rand_like(tensor) + min_val


def sample_from_discretized_mix_logistic(y, img_channels=3, log_scale_min=-7.):
    """

    :param y: Tensor, shape=(batch_size, 3 * num_mixtures * img_channels, height, width),
    :return: Tensor: sample in range of [-1, 1]
    """

    # unpack parameters, [batch_size, num_mixtures * img_channels, height, width] x 3
    logit_probs, means, log_scales = y.chunk(3, dim=1)

    temp = random_uniform_like(logit_probs, min_val=1e-5, max_val=1. - 1e-5)
    temp = logit_probs - torch.log(-torch.log(temp))

    ones = torch.eye(means.size(1) // img_channels, dtype=means.dtype, device=means.device)

    sample = []
    for logit_prob, mean, log_scale, tmp in zip(logit_probs.chunk(img_channels, dim=1),
                                                means.chunk(img_channels, dim=1),
                                                log_scales.chunk(img_channels, dim=1),
                                                temp.chunk(img_channels, dim=1)):
        # (batch_size, height, width)
        argmax = torch.max(tmp, dim=1)[1]
        B, H, W = argmax.shape

        one_hot = ones.index_select(0, argmax.flatten())
        one_hot = one_hot.view(B, H, W, mean.size(1)).permute(0, 3, 1, 2).contiguous()

        # (batch_size, 1, height, width)
        mean = torch.sum(mean * one_hot, dim=1)
        log_scale = torch.clamp_max(torch.sum(log_scale * one_hot, dim=1), log_scale_min)

        u = random_uniform_like(mean, min_val=1e-5, max_val=1. - 1e-5)
        x = mean + torch.exp(log_scale) * (torch.log(u) - torch.log(1 - u))
        sample.append(x)

    # (batch_size, img_channels, height, width)
    sample = torch.stack(sample, dim=1)

    return sample

class get_dataset(torch.utils.data.Dataset):

        ''' create a dataset suitable for pytorch models'''

        def __init__(self, x, device):
            self.x = torch.tensor(x, dtype=torch.float32).to(device)

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, index):
            return self.x[index, :, :, :]


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model,sigma=1,dt=.1,max_length=10,init_ind=0,data_dim=64,model_name='non', device='cpu'):
        super().__init__()
        self.model = model
        self.model_name=model_name
        self.sigma = sigma
        self.device = device
        self.dt=dt
        self.max_length = max_length
        self.init_ind = torch.tensor([init_ind]).to(device)
        self.data_dim=data_dim



    # Drift
    def f(self, t, y):
        y = y.view(-1, 1, self.data_dim, self.data_dim)


        predicts = self.model(t, torch.cat((y, y), dim=1),
                              y=self.init_ind * torch.ones((y.shape[0], )).int().to(self.device))
        if ((t== 0) ):

            self.x0 =y
            self.eps_t1 = predicts[:, 1:, :, :]
            outs = (predicts[:, :1, :, :]-self.x0)
        elif ((t== 1) ):
            outs = (predicts[:, :1, :, :]-y)/self.dt
        else:
            #outs = (predicts[:, :1, :, :]-y)/(1-t)+(predicts[:, 1:, :, :]/2) *1*torch.sqrt((t)*(1-t))
            #outs =torch.sqrt(1-t*(1-t))*(predicts[:, :1, :, :]  - y) / (1 - t) + (predicts[:, 1:, :, :] / 2) * 1 * torch.sqrt((t) * (1 - t))
            outs =torch.sqrt(1-(self.sigma**2)*t*(1-t))*(predicts[:, :1, :, :]  - y) / (1 - t)\
                  - (predicts[:, 1:, :, :]) * self.sigma * torch.sqrt((t) * (1 - t))

        return outs.flatten(start_dim=1)

    # Diffusion
    def g(self, t, y):

        return torch.ones_like(y) *self.sigma*0*torch.sqrt((t)*(1-t))
def generate_samples(model, flags, savedir, step, net_="normal",sde_enable=False,sigma=.01,model_name='non'):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if flags.parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    if sde_enable:
        with torch.no_grad():
            y_hat = torch.zeros((flags.N_steps,flags.batch_size, 1,  flags.data_dim , flags.data_dim )).to(device).float()
            for tt in range(flags.N_steps):

                    sde = SDE(model, model_name="bb", init_ind=tt,
                              max_length=flags.N_steps, data_dim=flags.data_dim,
                              sigma=0 * sigma, device=device)
                    if tt == 0:
                        x1 =  torch.randn(flags.batch_size, 1 * flags.data_dim * flags.data_dim, device=device).to(device).float()
                    else:
                        x1 = (y_hat[tt - 1]).view(flags.batch_size,1 * flags.data_dim * flags.data_dim).to(
                            device).float()  # +1*scale*torch.randn((trjs_s,y_hat.shape[-1])).to(device).float()

                    traj = torchsde.sdeint(
                        sde,
                        # x0.view(x0.size(0), -1),
                        x1,
                        ts=torch.linspace(0, 1, 3, device=device),
                        dt=.33,
                    )
                    y_hat[tt] = traj.cpu()[-1].view([-1, 1, flags.data_dim, flags.data_dim])



        images_to_show = y_hat.clip(-1, 1).squeeze()
        images_to_show = images_to_show / 2 + 0.5
        save_image(images_to_show.view(-1,1,flags.data_dim,flags.data_dim), savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=flags.N_steps)
    else:
        pass

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


def get_crps(im1,im2,observation_size,number_samples):
    observations = xr.DataArray(im1,
                                coords=[('sample', np.arange(im1.shape[0])),
                                        ('x', np.arange(observation_size)),
                                        ('y', np.arange(observation_size))

                                        ])
    forecasts = xr.DataArray(im2,
                             coords=[('sample', np.arange(im2.shape[0])),
                                     ('x', np.arange(observation_size)),
                                     ('y', np.arange(observation_size)),
                                     ('member', np.arange(number_samples))])
    crp = xskillscore.crps_ensemble(observations,forecasts)
    return crp
