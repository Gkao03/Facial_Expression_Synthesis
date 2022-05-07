import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import *
from helper_data import *
from helper_plotting import *
from torch.utils.data import Dataset, DataLoader
import cv2

''' 
Targets are 40-dim vectors representing
    00 - 5_o_Clock_Shadow
    01 - Arched_Eyebrows
    02 - Attractive 
    03 - Bags_Under_Eyes
    04 - Bald
    05 - Bangs
    06 - Big_Lips
    07 - Big_Nose
    08 - Black_Hair
    09 - Blond_Hair
    10 - Blurry 
    11 - Brown_Hair 
    12 - Bushy_Eyebrows 
    13 - Chubby 
    14 - Double_Chin 
    15 - Eyeglasses 
    16 - Goatee 
    17 - Gray_Hair 
    18 - Heavy_Makeup 
    19 - High_Cheekbones 
    20 - Male 
    21 - Mouth_Slightly_Open 
    22 - Mustache 
    23 - Narrow_Eyes 
    24 - No_Beard 
    25 - Oval_Face 
    26 - Pale_Skin 
    27 - Pointy_Nose 
    28 - Receding_Hairline 
    29 - Rosy_Cheeks 
    30 - Sideburns 
    31 - Smiling 
    32 - Straight_Hair 
    33 - Wavy_Hair 
    34 - Wearing_Earrings 
    35 - Wearing_Hat 
    36 - Wearing_Lipstick 
    37 - Wearing_Necklace 
    38 - Wearing_Necktie 
    39 - Young         
'''


''' Given CelebA dataset, return 
pass in 
attrs: binary attributes (1 or 0)
filenames: list of filenames corresponding to images 

X:  Image batch dimensions: torch.Size([5000, 3, 128, 128])
y:  Image label dimensions: torch.Size([5000, 40])
'''
class CelebADataset(Dataset):
  
    def __init__(self,  root_dir,
                        attrs,
                        filenames,
                        transforms=None):
        '''
         initilaize the important passed variables
        '''
        self.root_dir=root_dir
        self.transforms=transforms
        self.attrs = attrs
        self.filenames = filenames
        
    def __len__(self):
        '''
        returns the no of dataset
        '''
        return len(self.filenames)
    
    def __getitem__(self, index):
        '''
        returns image in numpy with labels
        '''

        # data/celeba/
        img_path = os.path.join(self.root_dir, 'img_align_celeba', self.filenames[index])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # select <index> row
        target = np.array(self.attrs[index,:])
        target[target==-1] = 0
    
        if self.transforms:
          image = self.transforms(image)

        return (image,target)

