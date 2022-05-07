import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import *
from helper_data import *
from helper_plotting import *
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from dataset import CelebADataset


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in train_loader:
        data = load_batch(batch_idx, True)
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (len(train_loader)*128),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader)*128)))
    return train_loss / (len(train_loader)*128)