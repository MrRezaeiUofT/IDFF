# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong
import os


import copy

import torch
from absl import app, flags

from utils_lsun import ema, generate_samples

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "IDFF", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_float("sigma", 0.2, help="sigma")
flags.DEFINE_float("flow_w", 2, help="flow weight")
# Training
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def sample_gen(argv=1):

    # MODELS
    net_model = UNetModelWrapper(
        dim=(6, 256, 256),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 1, 2, 2, 4, 4],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )  # new dropout + bs of 128
    sigma = FLAGS.sigma


    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    savedir =  FLAGS.output_dir + FLAGS.model +'-'+str(FLAGS.flow_w)+'-'+ str(sigma)+ "/"

    # Load the model
    PATH = f"{FLAGS.output_dir}/{FLAGS.model+'-'+str(FLAGS.flow_w)+'-'+ str(sigma)}/{FLAGS.model}_lsun_church_weights_step_final.pt"
    print("path: ", PATH)
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    state_dict = checkpoint["ema_model"]
    try:
        net_model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        net_model.load_state_dict(new_state_dict)
    net_model.eval()
    ema_model = copy.deepcopy(net_model)
    ema(net_model, ema_model, FLAGS.ema_decay)  # new

    # generate_samples(net_model, FLAGS.parallel, savedir, FLAGS.step, net_="normal",sde_enable=True,sigma=0*sigma,model_name=FLAGS.model)
    generate_samples(ema_model, FLAGS.parallel, savedir, 'final', net_="ema",sde_enable=True,sigma=1*sigma,model_name=FLAGS.model)



if __name__ == "__main__":
    app.run(sample_gen)
