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
##########################
### SETTINGS
##########################

# Device

DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# Hyperparameters
RANDOM_SEED = 123
BATCH_SIZE = 25

# 41 cols 
cols = range(1,41)

attrs = np.loadtxt('data/celeba/list_attr_celeba.txt', delimiter=',', skiprows=1, usecols=cols)
filenames = np.loadtxt('data/celeba/list_attr_celeba.txt', delimiter=',', skiprows=1, usecols=[0], dtype=str)
# should be 202599 x 41


##########################
### Dataset
##########################
# unzip_data('./data/celeba/img_align_celeba.zip', './data/celeba')

# exit(-1)
custom_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.CenterCrop((128, 128)),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataroot = './data/celeba/'

dataset = CelebADataset(root_dir=dataroot,
                        attrs=attrs,
                        filenames=filenames,
                        transforms=custom_transforms)


down  = list(range(0, len(dataset), 100))
dataset = torch.utils.data.Subset(dataset, down)

train_size = int(len(dataset) * 0.6)
valid_size = int(len(dataset) * 0.2)
test_size = len(dataset) - train_size - valid_size

# dataset = torchvision.datasets.ImageFolder(root=dataroot,
#                            transform=custom_transforms)
# 162770,19867,19962
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])


print('train dataset size', len(train_dataset))
print('valid dataset size', len(valid_dataset))
print('test dataset size', len(test_dataset))

# Create the dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

torch.manual_seed(RANDOM_SEED)
for idx, (images, labels) in enumerate(train_loader):
    # labels = lab[]
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    # print(labels[5])
    break

    


## 1) Image Manipulation in Original Space

# Compute Average Faces
IMG_IDX = 2
SRC_IMAGE = images[IMG_IDX]
FEATURES = {'smile': 31}


# for feat_name, feat_idx in FEATURES.items():
#     print('feat_name', feat_name, 'feat_idx', feat_idx)
#     print(f'**** Computing avg face with, without {feat_name} ****** ')

#     avg_img_with_feat, avg_img_without_feat = compute_average_faces(
#         feature_idx=feat_idx,
#         image_dim=(3, 128, 128),
#         data_loader=train_loader,
#         device=None,
#         encoding_fn=None)
#     print(f'**** Done computing avg face with, without {feat_name} ****** ')

#     # average face with feature
#     fig, ax = plt.subplots(figsize=(2, 2))
#     ax.imshow((avg_img_with_feat).permute(1, 2, 0))
#     plt.show()
#     plt.savefig(f'outputs/{IMG_IDX}_avg_{feat_name}.jpg')

#     # average face without feature 
#     fig, ax = plt.subplots(figsize=(2, 2))
#     ax.imshow((avg_img_without_feat).permute(1, 2, 0))
#     plt.show()
#     plt.savefig(f'outputs/{IMG_IDX}_avg_no_{feat_name}.jpg')

#     # original image 
#     print(f'**** Original image {IMG_IDX} ****** ')
#     fig, ax = plt.subplots(figsize=(2, 2))
#     ax.imshow(SRC_IMAGE.permute(1, 2, 0))
#     plt.show()
#     plt.savefig(f'outputs/{IMG_IDX}.jpg')

#     diff = (avg_img_with_feat - avg_img_without_feat)
#     plot_modified_faces(original=SRC_IMAGE,
#                         diff=diff)

#     # take difference in positive feature, negative feature image
#     print(f'**** Computing difference image  ****** ')

#     plt.tight_layout()
#     plt.show()
#     plt.savefig(f'outputs/{IMG_IDX}_diff_{feat_name}.jpg')

model = VAE()
model.to(DEVICE)
model.train()
train_loss = 0



reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
def loss_function(x,  mu, logvar, recon_x):
    BCE = reconstruction_function(recon_x, x)
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element).mul_(-0.5)

    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

max_epochs = 25
for epoch in range(max_epochs): 
    print(f'********* Epoch {epoch}**********')
    epoch_loss = []
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()

        encoded, z_mean, z_log_var, decoded = model(images)
        # print(encoded.shape, z_mean.shape, z_log_var.shape, decoded.shape)
        batch_loss = loss_function(images, z_mean, z_log_var, decoded)
        batch_loss.backward()

        epoch_loss.append(batch_loss.item())
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Loss', batch_loss.item())

    print(f'====> Epoch: {epoch} Average loss: {np.mean(epoch_loss)}')


torch.save(model, 'vae_model.pt')






exit(-1)

''' 2) Image Manipulation in Latent Space'''

# load VAE model 
model = VAE()
model.load_state_dict(torch.load('vae_celeba_02.pt', map_location=torch.device('cpu')))
model.to(DEVICE)

#  Compute Average Faces in Latent Space 

avg_img_with_feat, avg_img_without_feat = compute_average_faces(
    feature_idx=31, # smiling
    image_dim=200,
    data_loader=train_loader,
    device=DEVICE,
    encoding_fn=model.encoding_fn)

diff = (avg_img_with_feat - avg_img_without_feat)

example_img = EXAMPLE_IMAGE.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    encoded = model.encoding_fn(example_img).squeeze(0).to('cpu')

plot_modified_faces(original=encoded,
                    decoding_fn=model.decoder,
                    device=DEVICE,
                    diff=diff)

plt.tight_layout()
plt.show()

"""### Compute Average Faces in Latent Space -- With or Without Glasses"""

avg_img_with_feat, avg_img_without_feat = compute_average_faces(
    feature_idx=15, # eyeglasses
    image_dim=200,
    data_loader=train_loader,
    device=DEVICE,
    encoding_fn=model.encoding_fn)

diff = (avg_img_with_feat - avg_img_without_feat)

example_img = EXAMPLE_IMAGE.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    encoded = model.encoding_fn(example_img).squeeze(0).to('cpu')

plot_modified_faces(original=encoded,
                    decoding_fn=model.decoder,
                    device=DEVICE,
                    diff=diff)

plt.tight_layout()
plt.show()