import torch
import torch.optim as optim
import multiprocessing
import time
import preprocess as prep
import numpy as np
import models
import utils
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from analyze import *
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



USE_CUDA = True
MODEL = 'dfc-300'
MODEL_PATH = './checkpoints/' + MODEL
LOG_PATH = './logs/' + MODEL + '/log.pkl'
OUTPUT_PATH = './samples/'
PLOT_PATH = './plots/' + MODEL
LATENT_SIZE = 100

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
model = models.BetaVAE(latent_size=LATENT_SIZE).to(device)
# model = models.DFCVAE(latent_size=LATENT_SIZE).to(device)
print('latent size:', model.latent_size)

attr_map, id_attr_map = prep.get_attributes()

if __name__ == "__main__":

    model.load_last_model(MODEL_PATH)

    '''
    generate images using model
    '''
    # samples = generate(model, 60, device)
    # save_image(samples, OUTPUT_PATH + MODEL + '.png', padding=0, nrow=10)

    # train_losses, test_losses = utils.read_log(LOG_PATH, ([], []))
    # plot_loss(train_losses, test_losses, PLOT_PATH)

    '''
    get image ids with corresponding attribute
    '''

    print('getting imgs with attribute eyeglasses ')
    # ims: list of 20 (3,64,64) images
    # im_ids: list of 20 im_ids

    a1 = "black hair"
    a2 = "young"

    a1_ims, a1_im_ids = get_attr_ims(a2, num=20)
    a2_ims, a2_im_ids =  get_attr_ims(a2, num=20)


    man_ids = ['056224.jpg', '118398.jpg', '168342.jpg']
    woman_ids = ['034343.jpg', '066393.jpg']

    man_a1_im_ids = a1_im_ids[:5] # man with attribute a1
    man = prep.get_ims(man_ids)     # man without attribute a1
    
    woman_a2_im_ids = a2_im_ids[:5]# woman with attribute a2
    woman = prep.get_ims(woman_ids) # woman without attribute a2


    man_a1 = prep.get_ims(man_a1_im_ids)
    man = prep.get_ims(man_ids)
    woman_a2 = prep.get_ims(woman_a2_im_ids)
    woman = prep.get_ims(woman_ids)

    # utils.show_images(man_sunglasses, tensor=True)
    # utils.show_images(man, tensor=True)
    # utils.show_images(woman_smiles, tensor=True)
    # utils.show_images(woman, tensor=True)

    '''
    latent arithmetic
    '''
    print('performing latent arithmetic')

    man_z = get_z(man[0], model, device)
    woman_z = get_z(woman[1], model, device)
    a1_z = get_average_z(man_a1, model, device) - get_average_z(man, model, device)
    arith1 = latent_arithmetic(man_z, sunglass_z, model, device)
    arith2 = latent_arithmetic(woman_z, sunglass_z, model, device)

    save_image(arith1 + arith2, OUTPUT_PATH + 'arithmetic-dfc-V2' + '.png', padding=0, nrow=10)

    '''
    linear interpolate
    '''
    print('perform linear interpolation ')

    inter1 = linear_interpolate(man[0], man[1], model, device)
    inter2 = linear_interpolate(woman[0], woman_a2[1], model, device)
    inter3 = linear_interpolate(woman[1], woman_a2[0], model, device)

    save_image(inter1 + inter2 + inter3, OUTPUT_PATH + 'interpolate-dfc-V2' + '.png', padding=0, nrow=10)
