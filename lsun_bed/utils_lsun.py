import copy

import torch
from torchdyn.core import NeuralODE
from torchvision.transforms import ToPILImage
import torchsde
# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
IMG_DIM=256
class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model,sigma=1,dt=.1,model_name='non', reverse=False):
        super().__init__()
        self.model = model
        self.model_name=model_name
        self.sigma = sigma
        self.reverse = reverse
        self.dt=dt



    # Drift
    def f(self, t, y):
        y = y.view(-1, 3, IMG_DIM, IMG_DIM)


        if self.reverse:
            t = 1 - t
            predicts = self.model(t, torch.cat((y, y), dim=1))
            outs = -predicts[:, :3, :, :] + predicts[:, 3:, :, :]
            return outs.flatten(start_dim=1)
        predicts = self.model(t, torch.cat((y, y), dim=1))
        if ((t== 0) ):

            self.x0 =y
            self.eps_t1 = predicts[:, 3:, :, :]
            outs = (predicts[:, :3, :, :]-self.x0)
        elif ((t== 1) ):
            outs = (predicts[:, :3, :, :]-y)/self.dt
        else:
            outs =(1-(self.sigma**2)*t*(1-t))*(predicts[:, :3, :, :]  - y) / (1 - t)\
                  - (predicts[:, 3:, :, :]) *  (self.sigma **2)* torch.sqrt((t) * (1 - t))

        return outs.flatten(start_dim=1)

    # Diffusion
    def g(self, t, y):

      return torch.ones_like(y) *self.sigma*0*torch.sqrt((t)*(1-t))

def generate_samples(model, parallel, savedir, step, net_="normal",sde_enable=False,sigma=.01,model_name='non'):
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
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    if sde_enable:
        sde = SDE(model,dt=.1, reverse=False,model_name=model_name, sigma=sigma)
        with torch.no_grad():
            sde_traj = torchsde.sdeint(
                sde,
                # x0.view(x0.size(0), -1),
                torch.randn(16, 3 * IMG_DIM * IMG_DIM, device=device),
                ts=torch.linspace(0, 1, 10, device=device),
                dt=sde.dt,
            )
        images_to_show = sde_traj[-1, :].view([-1, 3, IMG_DIM, IMG_DIM]).clip(-1, 1)
        images_to_show = images_to_show / 2 + 0.5
        save_image(images_to_show, savedir + f"{net_}_generated_IDFF_images_step_{step}.png", nrow=4)
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
        for x, y in iter(dataloader):
            yield x

# Define LightningDataModule

