import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from models import *
from helper_data import *
from helper_plotting import *
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from dataset import CelebADataset

model = VAE()
model.load_state_dict(torch.load('vae_celeba_02.pt', map_location=torch.device('cpu')))
model.to(DEVICE)

model.test()
for batch_idx, (images, labels) in enumerate(test_loader):
    images, labels = images.to(DEVICE), labels.to(DEVICE)
  
    encoded, z_mean, z_log_var, decoded = model(images)
   
    plt.imshow(images[0])
    plt.savefig(f'{batch_idx}_orig.jpg')

    plt.imshow(decoded[0])
    plt.savefig(f'{batch_idx}_recon.jpg'