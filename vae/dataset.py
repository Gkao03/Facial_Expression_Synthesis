import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import *
from helper_data import *
from helper_plotting import *
from torch.utils.data import Dataset, DataLoader



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

