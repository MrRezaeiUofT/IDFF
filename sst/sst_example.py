# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import ssl
import pickle
from torch.utils.data import DataLoader
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from collections import OrderedDict
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange
from utils import ema, generate_samples, infiniteloop,get_dataset

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher, SchrodingerBridgeConditionalFlowMatcher
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "IDFF", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_float("sigma", 0.2, help="sigma")
flags.DEFINE_float("flow_w", 2, help="flow weight")
# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 800001, help="total training steps"
)  # Lipman et al uses 500k but double batch sizec
flags.DEFINE_integer(
    "pre_step", 870000, help="total training steps")
flags.DEFINE_bool("pretrain", False, help="enable pre-train")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 8, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("data_dim", 60, help="data size")  # Lipman et al uses 128
flags.DEFINE_integer("N_steps", 10, help="data size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    100000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    with open('data/SST_dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)
    FLAGS.N_steps = dataset['N_steps']


    x_tr = dataset['x_tr']
    train_dataset = get_dataset(x_tr, device)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    datalooper = infiniteloop(train_loader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(2, FLAGS.data_dim, FLAGS.data_dim),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 4],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        class_cond=True,
    ).to(
        device
    )  # new dropout + bs of 128
    if FLAGS.pretrain:
        PATH_pre = f"{FLAGS.output_dir}/{FLAGS.model + '-' + str(FLAGS.flow_w) + '-' + str(FLAGS.sigma)}/{FLAGS.model}_sst_weights_step_{FLAGS.pre_step}.pt"
        print("path: ", PATH_pre)
        checkpoint = torch.load(PATH_pre)
        state_dict = checkpoint["ema_model"]
        try:
            net_model.load_state_dict(state_dict)
            print('pretrained model loaded')
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            net_model.load_state_dict(new_state_dict)
    else:
        print('No-pretrained model loaded')

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.AdamW(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = FLAGS.sigma
    print('siigma=%f' % (sigma))

    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)


    savedir = FLAGS.output_dir + FLAGS.model + '-' + str(FLAGS.flow_w) + '-' + str(sigma) + "/"
    os.makedirs(savedir, exist_ok=True)
    
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            index_sel = torch.randint(1, FLAGS.N_steps , (FLAGS.batch_size,)).to(device)
            idss=torch.ones((FLAGS.batch_size,FLAGS.batch_size)).to(device)
            idss=(idss*index_sel).int().view(-1,)
            data_in = next(datalooper).to(device)
            data_in=2*(data_in-.5)
            x1=data_in[:,index_sel].view(-1,1,FLAGS.data_dim,FLAGS.data_dim)
            x0 = data_in[:,index_sel-1].view(-1,1,FLAGS.data_dim,FLAGS.data_dim)

            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
            lambda_t = FM.compute_lambda(t)

            outs = net_model(t, torch.cat((xt, xt), dim=1),
                             y=idss)
            vt = outs[:, :1, :, :]
            st = outs[:, 1:, :, :]

            flow_loss = torch.mean( ((vt - FM.x1) ** 2))
            score_loss = torch.mean((lambda_t[:, None, None, None] * st + eps) ** 2)
            loss = FLAGS.flow_w * flow_loss + score_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                if FLAGS.pretrain:
                    generate_samples(net_model, FLAGS, savedir, step + FLAGS.pre_step, net_="normal",
                                     sde_enable=True, sigma=sigma, model_name=FLAGS.model)
                    generate_samples(ema_model, FLAGS, savedir, step + FLAGS.pre_step, net_="ema",
                                     sde_enable=True, sigma=sigma, model_name=FLAGS.model)
                    torch.save(
                        {
                            "net_model": net_model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "sched": sched.state_dict(),
                            "optim": optim.state_dict(),
                            "step": step + FLAGS.pre_step,
                        },
                        savedir + f"{FLAGS.model}_sst_weights_step_{step + FLAGS.pre_step}.pt",
                    )
                else:
                    generate_samples(net_model, FLAGS, savedir, step, net_="normal", sde_enable=True,
                                     sigma=sigma, model_name=FLAGS.model)
                    # generate_samples(ema_model, FLAGS, savedir, step, net_="ema", sde_enable=True, sigma=sigma,
                    #                  model_name=FLAGS.model)
                    torch.save(
                        {
                            "net_model": net_model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "sched": sched.state_dict(),
                            "optim": optim.state_dict(),
                            "step": step,
                        },
                        savedir + f"{FLAGS.model}_sst_weights_step_{step}.pt",
                    )


if __name__ == "__main__":
    app.run(train)