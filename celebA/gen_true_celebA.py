import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import make_grid, save_image
seed_value = 73
torch.manual_seed(seed_value)
import matplotlib.pyplot as plt
import numpy as np
# Set the seed for NumPy if you are using it with PyTorch
np.random.seed(seed_value)

# Hyperparameters
batch_size=500
total_num=50000
device = torch.device('cpu')
transform = transforms.Compose([transforms.Resize(64),
            transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
train_dataset = torchvision.datasets.CelebA(root='../../alternators/data', split='train', transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

from PIL import Image
save_dir='./datasets/celebA64/all_images/'
def save_figs(images, init_label):
    for ii in range(images.shape[0]):

        # Save the PIL image to a file
        file_path =init_label+"_imes_"+str(ii)+'.png'
        Image.fromarray(images[ii], 'RGB').save(file_path)

for i, data in enumerate(train_loader, 0):
    print("%d,%d"%(i,total_num/batch_size))
    if i >= total_num/batch_size:
        break
    inputs, labels = data
    inputs = (inputs * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    save_figs(inputs,save_dir+str(i*100))


