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
from helpers import *
import os
import cv2

USE_CUDA = True
MODEL = 'dfc-300'
MODEL_PATH = './checkpoints/' + MODEL
LOG_PATH = './logs/' + MODEL + '/log.pkl'
OUTPUT_PATH = f'./{MODEL}_outputs/'
PLOT_PATH = './plots/' + MODEL
LATENT_SIZE = 100
IMAGE_PATH = '../../data/celeba/img_align_celeba/'

RUN = 1



''' Performs Latent Exploration using Beta-VAE '''

class LatentExploration():

    # Initialize model, attributes 
    def __init__(self, model, 
                       attrA,
                       attrB,
                       genderA,
                       genderB,
                       save_path,
                       device):
        self.model = model 

        # attributes 
        self.attrA = attrA
        self.attrB= attrB

        self.genderA = genderA
        self.genderB = genderB

        self.device = device
        self.attr_map, self.id_attr_map = prep.get_attributes()

        # output path 
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def test(self):
        # maps attr names to integer: {'5_o_clock_shadow': 0, 'arched_eyebrows': 1, 'attractive': 2, 'bags_under_eyes': 3..}
        print(self.attr_map)
    #     # maps attr name to 40-dim binary vector: {'5_o_clock_shadow':[-1,1,1;.]}
    #     print(self.id_attr_map.shape)

    # Retrieve image ids ("xxx.jpg")
    
    def get_ids(self):
        # ids corresponding to specified gender with attribute A
        _, with_A_ids = get_attr_ims(attr=self.attrA, num=10, has=True,gender=self.genderA)
        # ids corresponding to specified gender without attribute A
        _, without_A_ids = get_attr_ims(attr=self.attrA, num=10, has=False,gender=self.genderA)
        _, with_B_ids = get_attr_ims(attr=self.attrB, num=10, has=True,gender=self.genderB)
        _, without_B_ids = get_attr_ims(attr=self.attrB, num=10, has=False,gender=self.genderB)

        # with_A_ids = ['172624.jpg', '164754.jpg', '089604.jpg', '024726.jpg']
        # without_A_ids = ['056224.jpg', '118398.jpg', '168342.jpg']
        # with_B_ids = ['168124.jpg', '176294.jpg', '169359.jpg']
        # without_B_ids = ['034343.jpg', '066393.jpg']

        return {"with_A": with_A_ids,
                "without_A": without_A_ids,
                "with_B":with_B_ids,
                "without_B":without_B_ids}

    # Retrieve images from ids
    def get_ims_from_ids(self, ids):
        self.with_A = prep.get_ims(ids['with_A'])
        self.without_A = prep.get_ims(ids['without_A'])
        self.with_B = prep.get_ims(ids['with_B'])
        self.without_B = prep.get_ims(ids['without_B'])
        

        print('Saving images to... ', self.save_path)

        # permute to get HxWxC (64,64,3)
        cv2.imwrite(os.path.join(self.save_path,'with_A.jpg'),255*np.transpose(np.array(self.with_A[0]), (1,2,0))[:,:,::-1])
        cv2.imwrite(os.path.join(self.save_path,'without_A.jpg'),255*np.transpose(np.array(self.without_A[0]), (1,2,0))[:,:,::-1])
        cv2.imwrite(os.path.join(self.save_path,'with_B.jpg'),255*np.transpose(np.array(self.with_B[0]), (1,2,0))[:,:,::-1])
        cv2.imwrite(os.path.join(self.save_path,'without_B.jpg'),255*np.transpose(np.array(self.without_B[0]), (1,2,0))[:,:,::-1])


        


    # need 1,3,64,64
    def do_latent_arithmetic(self):
        z_without_A  = get_z(self.without_A[0], self.model, self.device)
        z_without_B = get_z(self.without_B[1], self.model, self.device)
        # get avg across with attrA - avg across without attrA
        z_avg = get_average_z(self.with_A, self.model, self.device) - get_average_z(self.without_A, self.model, self.device)

        arith1 = latent_arithmetic(z_without_A, z_avg, self.model, self.device)
        arith2 = latent_arithmetic(z_without_B, z_avg, self.model, self.device)
        

        print('Saving latent arithmetic output')
        save_image(arith1 + arith2, os.path.join(self.save_path, 'latent_arith.png'), padding=0, nrow=10)
        return arith1 + arith2 


    def do_linear_interpolation(self):

        inter1 = linear_interpolate(self.without_A[0], self.without_A[1], self.model, self.device)
        inter2 = linear_interpolate(self.without_B[0], self.with_B[1], self.model, self.device)
        inter3 = linear_interpolate(self.without_B[1], self.with_B[0], self.model, self.device)

        save_image(inter1 + inter2 + inter3, os.path.join(self.save_path, 'latent_interp.png'), padding=0, nrow=10)



def main():

    '''
    Available Attributes:

    {'5_o_clock_shadow': 0, 'arched_eyebrows': 1, 'attractive': 2, 'bags_under_eyes': 3, 'bald': 4,
    'bangs': 5, 'big_lips': 6, 'big_nose': 7, 'black_hair': 8, 'blond_hair': 9, 'blurry': 10,
    'brown_hair': 11, 'bushy_eyebrows': 12, 'chubby': 13, 'double_chin': 14, 'eyeglasses': 15,
    'goatee': 16, 'gray_hair': 17, 'heavy_makeup': 18, 'high_cheekbones': 19, 'male': 20, 'mouth_slightly_open': 21,
    'mustache': 22, 'narrow_eyes': 23, 'no_beard': 24, 'oval_face': 25, 'pale_skin': 26, 'pointy_nose': 27, 
    'receding_hairline': 28, 'rosy_cheeks': 29, 'sideburns': 30, 'smiling': 31, 'straight_hair': 32,
    'wavy_hair': 33, 'wearing_earrings': 34, 'wearing_hat': 35, 'wearing_lipstick': 36, 'wearing_necklace': 37, 
    'wearing_necktie': 38, 'young': 39}
    '''

    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    # model = models.BetaVAE(latent_size=LATENT_SIZE).to(device)
    model = models.DFCVAE(latent_size=LATENT_SIZE).to(device)
    model.load_last_model(MODEL_PATH)
    print('latent size:', model.latent_size)
    attr_map, id_attr_map = prep.get_attributes()

    train_losses, test_losses = utils.read_log(LOG_PATH, ([], []))
    plot_loss(train_losses, test_losses, PLOT_PATH)





    attrA = "wearing_necktie"
    attrB = "straight_hair"
    genderA = "male"
    genderB = "female"
    SAVE_PATH = os.path.join(OUTPUT_PATH, f'run_{genderA}-{attrA}_{genderB}-{attrB}')

    exp = LatentExploration(model, attrA, attrB, genderA, genderB, SAVE_PATH, device)
    ids = exp.get_ids()
    exp.get_ims_from_ids(ids)
    exp.do_latent_arithmetic()
    exp.do_linear_interpolation()






if __name__ == "__main__":

    main()

    


    # '''
    # get image ids with corresponding attribute
    # '''

    # print('getting imgs with attribute eyeglasses ')
    # # ims: list of 20 (3,64,64) images
    # # im_ids: list of 20 im_ids

    # a1 = "black hair"
    # a2 = "young"

    # a1_ims, a1_im_ids = get_attr_ims(a2, num=20)
    # a2_ims, a2_im_ids =  get_attr_ims(a2, num=20)


    # man_ids = ['056224.jpg', '118398.jpg', '168342.jpg']
    # woman_ids = ['034343.jpg', '066393.jpg']

    # man_a1_im_ids = a1_im_ids[:5] # man with attribute a1
    # man = prep.get_ims(man_ids)     # man without attribute a1
    
    # woman_a2_im_ids = a2_im_ids[:5]# woman with attribute a2
    # woman = prep.get_ims(woman_ids) # woman without attribute a2


    # man_a1 = prep.get_ims(man_a1_im_ids)
    # man = prep.get_ims(man_ids)
    # woman_a2 = prep.get_ims(woman_a2_im_ids)
    # woman = prep.get_ims(woman_ids)

    # # utils.show_images(man_sunglasses, tensor=True)
    # # utils.show_images(man, tensor=True)
    # # utils.show_images(woman_smiles, tensor=True)
    # # utils.show_images(woman, tensor=True)

    # '''
    # latent arithmetic
    # '''
    # print('performing latent arithmetic')

    # man_z = get_z(man[0], model, device)
    # woman_z = get_z(woman[1], model, device)
    # a1_z = get_average_z(man_a1, model, device) - get_average_z(man, model, device)
    # arith1 = latent_arithmetic(man_z, sunglass_z, model, device)
    # arith2 = latent_arithmetic(woman_z, sunglass_z, model, device)

    # save_image(arith1 + arith2, OUTPUT_PATH + 'arithmetic-dfc-V2' + '.png', padding=0, nrow=10)

    # '''
    # linear interpolate
    # '''
    # print('perform linear interpolation ')

    # inter1 = linear_interpolate(man[0], man[1], model, device)
    # inter2 = linear_interpolate(woman[0], woman_a2[1], model, device)
    # inter3 = linear_interpolate(woman[1], woman_a2[0], model, device)

    # save_image(inter1 + inter2 + inter3, OUTPUT_PATH + 'interpolate-dfc-V2' + '.png', padding=0, nrow=10)
