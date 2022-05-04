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
##########################
### SETTINGS
##########################

# Device
CUDA_DEVICE_NUM = 3
DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# Hyperparameters
RANDOM_SEED = 123
BATCH_SIZE = 500

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


# dataset = torchvision.datasets.ImageFolder(root=dataroot,
#                            transform=custom_transforms)

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [162770,19867,19962])


print('train dataset size', len(train_dataset))
print('valid dataset size', len(valid_dataset))
print('test dataset size', len(test_dataset))

# Create the dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                         shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False)



# Image batch dimensions: torch.Size([5000, 3, 128, 128])
# Image label dimensions: torch.Size([5000, 40])
torch.manual_seed(RANDOM_SEED)
for idx, (images, labels) in enumerate(train_loader):
    # labels = lab[]
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[5])
    #print(labels[:10])
    break
    


## 1) Image Manipulation in Original Space

# Compute Average Faces

 
IMG_IDX = 2
SRC_IMAGE = images[IMG_IDX]
FEATURES = {'smile': 31}


for feat_name, feat_idx in FEATURES.items():
    print('feat_name', feat_name, 'feat_idx', feat_idx)
    print(f'**** Computing avg face with, without {feat_name} ****** ')

    avg_img_with_feat, avg_img_without_feat = compute_average_faces(
        feature_idx=feat_idx,
        image_dim=(3, 128, 128),
        data_loader=train_loader,
        device=None,
        encoding_fn=None)
    print(f'**** Done computing avg face with, without {feat_name} ****** ')

    # average face with feature
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow((avg_img_with_feat).permute(1, 2, 0))
    plt.show()
    plt.savefig(f'outputs/{IMG_IDX}_avg_{feat_name}.jpg')

    # average face without feature 
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow((avg_img_without_feat).permute(1, 2, 0))
    plt.show()
    plt.savefig(f'outputs/{IMG_IDX}_avg_no_{feat_name}.jpg')

    # original image 
    print(f'**** Original image {IMG_IDX} ****** ')
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(SRC_IMAGE.permute(1, 2, 0))
    plt.show()
    plt.savefig(f'outputs/{IMG_IDX}.jpg')

    diff = (avg_img_with_feat - avg_img_without_feat)
    plot_modified_faces(original=SRC_IMAGE,
                        diff=diff)

    # take difference in positive feature, negative feature image
    print(f'**** Computing difference image  ****** ')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'outputs/{IMG_IDX}_diff_{feat_name}.jpg')



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