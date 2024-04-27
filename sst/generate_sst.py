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
import torchsde
import numpy as np
from utils import get_dataset,SDE, ema
import time
import cv2
import matplotlib.pyplot as plt

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "IDFF", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_float("sigma", 0.2, help="sigma")
flags.DEFINE_float("flow_w", 2, help="flow weight")
# Training


flags.DEFINE_bool("pretrain", True, help="enable pre-train")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 2, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("data_dim", 60, help="data size")  # Lipman et al uses 128
flags.DEFINE_integer("N_steps", 10, help="data size")  # Lipman et al uses 128
flags.DEFINE_integer("number_samples", 1, help="")  # Lipman et al uses 128
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
        save_dir=f"{FLAGS.output_dir}/{FLAGS.model + '-' + str(FLAGS.flow_w) + '-' + str(FLAGS.sigma)}/{FLAGS.model}_sst_weights_step_final_"
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

    net_model = copy.deepcopy(net_model)
    ema_model = copy.deepcopy(net_model)
    ema(net_model, ema_model, FLAGS.ema_decay)
    ema_model.eval()
    
    
    
    for forcasting_h in range(1,10):
        for kk in range(4):
            SDESIGMA = kk* FLAGS.sigma
            with torch.no_grad():
                    # create mask
        
        
                    for sample_indx, data in enumerate(test_loader, 0):
                        print(sample_indx)
                        if sample_indx >10:
                           break
                        data = torch.unsqueeze(data, dim=2)
                        data=2*(data-.5)
                        inputs = torch.transpose(data, dim0=0, dim1=1)
                        x_te = data[:, -1 * forcasting_h:, :, :, :].squeeze()
        
                        y_hat = torch.zeros((FLAGS.N_steps, inputs.shape[1], 1, FLAGS.data_dim, FLAGS.data_dim)).to(
                                device).float()
                        y_hat[:-1*forcasting_h] =inputs[:-1*forcasting_h]
                        for tt in range(FLAGS.N_steps-forcasting_h,FLAGS.N_steps):
        
                                sde = SDE(net_model, model_name="bb", init_ind=tt,
                                          max_length=FLAGS.N_steps, data_dim=FLAGS.data_dim,
                                          sigma= SDESIGMA, device=device)
                                if tt == FLAGS.N_steps-forcasting_h:
                                    x1 = inputs[FLAGS.N_steps-forcasting_h-1].view(inputs.shape[1], 1 * FLAGS.data_dim * FLAGS.data_dim).to(
                                        device).float()

                                else:
                                    x1 =y_hat[tt - 1].view( inputs.shape[1], 1 * FLAGS.data_dim *FLAGS.data_dim).to(
                                        device).float() #+.5*torch.randn((x1.shape)).to(device).float()

                                traj = torchsde.sdeint(
                                    sde,
                                    # x0.view(x0.size(0), -1),
                                    x1,
                                    ts=torch.linspace(0, 1,5, device=device),
                                    dt=.2,
                                )
                                y_hat[tt] = traj.cpu()[-1].view([-1, 1, FLAGS.data_dim, FLAGS.data_dim])
        

        
                        images_to_show = y_hat.clip(-1, 1).squeeze()
                        images_to_show=(images_to_show-images_to_show.min())/(images_to_show.min()-images_to_show.max())
                        # images_to_show = images_to_show / 2 + 0.5
                        # images_to_show=torch.clamp(images_to_show, 0, 1)
                        images_to_show = images_to_show.permute(1, 0, 2, 3).cpu().numpy()
                        images_to_show=images_to_show[:, -1 * forcasting_h:,  :, :].squeeze()
        
                        x_te = (x_te - x_te.min()) / (x_te.min() - x_te.max())
                        # x_te = x_te / 2 + 0.5
                        # x_te=torch.clamp(x_te, 0, 1)
                        x_te = x_te.cpu().numpy()
                        fig, axs = plt.subplots(2, 1,figsize=(10, 3) )
        
                        for ii in range(images_to_show.shape[0]):
                            heatmaps = []
                            heatmaps_true = []
        
                            for i in range(images_to_show.shape[1]):
                                heatmap = cv2.applyColorMap(np.uint8(255 * images_to_show[ii, i].squeeze()), cv2.COLORMAP_JET)
                                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                heatmaps.append(heatmap)
        
                            # Concatenate the heatmaps horizontally
                            concatenated_heatmap = np.hstack(heatmaps)
        
                            # Save the concatenated heatmap image
                            axs[0].imshow(concatenated_heatmap,interpolation='gaussian')
                            # axs[0].set_ylabel('Predicted')
        
                            for i in range(images_to_show.shape[1]):
                                heatmap = cv2.applyColorMap(np.uint8(255 * x_te[ii, i].squeeze()), cv2.COLORMAP_JET)
                                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                heatmaps_true.append(heatmap)
        
                            # Concatenate the heatmaps horizontally
                            concatenated_heatmap_true = np.hstack(heatmaps_true)
        
                            # Save the concatenated heatmap image
                            axs[1].imshow(concatenated_heatmap_true,interpolation='gaussian')

                            plt.tight_layout()
                            for ax in axs:
                                ax.axis('off')
                                # ax.tight_layout()
                            plt.savefig(save_dir + f"net_{SDESIGMA:.1f}_{forcasting_h}_{sample_indx}_{ii}.png")
                        plt.close()







if __name__ == "__main__":
    app.run(train)


