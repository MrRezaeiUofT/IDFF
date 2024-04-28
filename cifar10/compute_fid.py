# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import os
import sys
import copy
import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from utils_cifar import SDE
from torchcfm.models.unet.unet import UNetModelWrapper
from utils_cifar import ema, generate_samples, infiniteloop
import torchsde
FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

flags.DEFINE_float("sigma", 0.2, help="sigma")
flags.DEFINE_float("flow_w", 2, help="flow weight")
# Training
flags.DEFINE_string("model", "IDFF", help="flow matching model type")
flags.DEFINE_string("input_dir", "./results", help="output_directory")

flags.DEFINE_integer("integration_steps", 10, help="number of inference steps")
flags.DEFINE_string("integration_method", "sde", help="integration method to use")
flags.DEFINE_integer("step",500000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_integer("batch_size", 64, help="")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
FLAGS(sys.argv)
SDE_sigma=1*FLAGS.sigma

for flag_name in FLAGS:
        flag_value = getattr(FLAGS, flag_name)
        print(f"{flag_name}: {flag_value}")
# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

new_net = UNetModelWrapper(
    dim=(6, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 4],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)


# Load the model
PATH = f"{FLAGS.input_dir}/{FLAGS.model+'-'+str(FLAGS.flow_w)+'-'+ str(FLAGS.sigma)}/{FLAGS.model}_cifar10_weights_step_final.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH)
state_dict = checkpoint["ema_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()
ema_model = copy.deepcopy(new_net)
ema(new_net, ema_model, FLAGS.ema_decay)
# Define the integration method if euler is used
if FLAGS.integration_method == "euler":
    node = NeuralODE(new_net, solver=FLAGS.integration_method)


def gen_1_img(unused_latent):
    with torch.no_grad():
        x = torch.randn(500, 3, 32, 32, device=device)
        if FLAGS.integration_method == "euler":
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
            traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
        else:
            print("Use method: ", FLAGS.integration_method)
            sde = SDE(ema_model, reverse=False, model_name=FLAGS.model, sigma=SDE_sigma)
            with torch.no_grad():
                sde_traj = torchsde.sdeint(
                    sde,
                    # x0.view(x0.size(0), -1),
                    torch.randn(500, 3 * 32 * 32, device=device),
                    ts=torch.linspace(0, 1, FLAGS.integration_steps, device=device),
                    dt= 1/FLAGS.integration_steps,
                )
            traj = sde_traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)

    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)#.permute(1, 2, 0)
    return img


print(f"sde sigma {SDE_sigma}")
print(f"sde dt { 1/FLAGS.integration_steps}")
print("Start computing FID")
score = fid.compute_fid(
    gen=gen_1_img,
    dataset_name="cifar10",
    batch_size=500,
    dataset_res=32,
    num_gen=FLAGS.num_gen,
    dataset_split="train",
    mode="legacy_pytorch",
)
print()
print("FID has been computed")
# print()
# print("Total NFE: ", new_net.nfe)
print()
print("FID: ", score)


