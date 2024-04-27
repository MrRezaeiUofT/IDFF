
import copy
import os
import ssl
import pickle
from torch.utils.data import DataLoader
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from collections import OrderedDict
from absl import app, flags
import torchsde
import numpy as np
from utils import ema, generate_samples, infiniteloop,get_dataset,get_crps,SDE
import time


from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "IDFF", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_float("sigma", 0.2, help="sigma")
flags.DEFINE_float("flow_w", 2, help="flow weight")
# Training

flags.DEFINE_integer(
    "pre_step", 1170000, help="total training steps")
flags.DEFINE_bool("pretrain", True, help="enable pre-train")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 32, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("data_dim", 60, help="data size")  # Lipman et al uses 128
flags.DEFINE_integer("N_steps", 10, help="data size")  # Lipman et al uses 128
flags.DEFINE_integer("number_samples", 20, help="")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 8, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):

    with open('data/SST_dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)
    FLAGS.N_steps = dataset['N_steps']

    x_te = dataset['x_te']
    scaling_term = dataset['max_x'] - dataset['min_x']
    test_dataset = get_dataset(x_te, device)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

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
        PATH_pre = f"{FLAGS.output_dir}/{FLAGS.model + '-' + str(FLAGS.flow_w) + '-' + str(FLAGS.sigma)}/{FLAGS.model}_sst_weights_step_final.pt"
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

    net_model.eval()
    ema_model = copy.deepcopy(net_model)
    ema(net_model, ema_model, FLAGS.ema_decay)
    for kk in range(4):
        SDESIGMA = kk * FLAGS.sigma
        print('ema-sigma=%f\n'%(SDESIGMA))
        SSR_all = []
        MSE_all = []
        CRPS_all = []
        net_model.eval()
        forcasting_h = [1,2,3,4,5,6,7]
    
        for fr in forcasting_h:
            print(fr)
            with torch.no_grad():
                # create mask
    
    
                for i, data in enumerate(test_loader, 0):
                    #print(i)
                    #if i >10:
                    #    break
                    data = torch.unsqueeze(data, dim=2)
                    data=2*(data-.5)
                    inputs = torch.transpose(data, dim0=0, dim1=1)
                    x_te = torch.transpose(inputs, dim0=0, dim1=1)[:, -1 * fr:, :, :, :].squeeze().reshape(-1,
                                                                                                           FLAGS.data_dim,
                                                                                                           FLAGS.data_dim)
                    x_te=x_te*(scaling_term)/2
                    for ss in range(FLAGS.number_samples):
                        y_hat = torch.zeros((FLAGS.N_steps, inputs.shape[1], 1, FLAGS.data_dim, FLAGS.data_dim)).to(
                            device).float()
                        y_hat[:-1*fr] =inputs[:-1*fr]
                        for tt in range(FLAGS.N_steps-fr,FLAGS.N_steps):
    
                            sde = SDE(ema_model, model_name="bb", init_ind=tt,
                                      max_length=FLAGS.N_steps, data_dim=FLAGS.data_dim,
                                      sigma=1 * FLAGS.sigma, device=device)
                            if tt == FLAGS.N_steps-fr:
                                x1 = inputs[FLAGS.N_steps-fr-1].view(inputs.shape[1], 1 * FLAGS.data_dim * FLAGS.data_dim).to(
                                    device).float()
                            else:
                                x1 = (y_hat[tt - 1]).view( inputs.shape[1], 1 * FLAGS.data_dim * FLAGS.data_dim).to(
                                    device).float()  # +1*scale*torch.randn((trjs_s,y_hat.shape[-1])).to(device).float()
    
                            traj = torchsde.sdeint(
                                sde,
                                # x0.view(x0.size(0), -1),
                                x1,
                                ts=torch.linspace(0, 1, 3, device=device),
                                dt=.2,
                            )
                            y_hat[tt] = traj.cpu()[-1].view([-1, 1, FLAGS.data_dim, FLAGS.data_dim])
    
                        x_hat_temp = y_hat
                        x_hat_temp=x_hat_temp*(scaling_term)/2
                        if ss == 0:
                                x_hat = torch.unsqueeze(
                                    torch.transpose(x_hat_temp, dim0=0, dim1=1)[:, -1 * fr:, :, :, :].squeeze().reshape(-1,
                                                                                                                        FLAGS.data_dim,
                                                                                                                        FLAGS.data_dim),
                                    dim=-1)
                        else:
                                temp = torch.unsqueeze(
                                    torch.transpose(x_hat_temp, dim0=0, dim1=1)[:, -1 * fr:, :, :, :].squeeze().reshape(-1,
                                                                                                                        FLAGS.data_dim,
                                                                                                                        FLAGS.data_dim),
                                    dim=-1)
                                x_hat = torch.cat((x_hat, temp), -1)
    
                    if i == 0:
                        x_hat_all = x_hat.detach().cpu().numpy().squeeze()
                        x_te_all = x_te.detach().cpu().numpy().squeeze()
                    else:
                        x_hat_all = np.concatenate([x_hat_all, x_hat.detach().cpu().numpy().squeeze()], axis=0)
                        x_te_all = np.concatenate([x_te_all, x_te.detach().cpu().numpy().squeeze()], axis=0)
                    MSE_all.append(np.mean((x_hat_all.mean(axis=-1) - x_te_all) ** 2))
                    spread = np.sqrt(np.var(x_hat_all.mean(axis=-1), axis=0).mean())
                    SSR_all.append(spread / np.sqrt(MSE_all[-1]))
                    CRPS_all.append(get_crps(x_te_all, x_hat_all,FLAGS.data_dim,FLAGS.number_samples))
    
        print('CRPs mean=%2.3f,std=%2.3f' % (np.array(CRPS_all).mean(), np.array(CRPS_all).std()))
        print('MSE mean=%2.3f,std=%2.3f' % (np.array(MSE_all).mean(), np.array(MSE_all).std()))
        print('SSR mean=%2.3f,std=%2.3f' % (np.array(SSR_all).mean(), np.array(SSR_all).std()))


if __name__ == "__main__":
    app.run(train)


