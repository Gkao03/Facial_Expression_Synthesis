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



# generate n=num images using the model
def generate(model, num, device):
    model.eval()
    z = torch.randn(num, model.latent_size).to(device)
    with torch.no_grad():
        return model.decode(z).cpu()


# returns pytorch tensor z
def get_z(im, model, device):
    model.eval()
    im = torch.unsqueeze(im, dim=0).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(im)
        z = model.sample(mu, logvar)

    return z


def linear_interpolate(im1, im2, model, device):
    model.eval()
    z1 = get_z(im1, model, device)
    z2 = get_z(im2, model, device)

    factors = np.linspace(1, 0, num=10)
    result = []

    with torch.no_grad():

        for f in factors:
            z = (f * z1 + (1 - f) * z2).to(device)
            im = torch.squeeze(model.decode(z).cpu())
            result.append(im)

    return result


def get_average_z(ims, model, device):
    model.eval()
    z = torch.unsqueeze(torch.zeros(model.latent_size), dim=0)

    for im in ims:
        z += get_z(im, model, device).cpu()

    return z / len(ims)


def latent_arithmetic(im_z, attr_z, model, device):
    model.eval()

    factors = np.linspace(0, 1, num=10, dtype=float)
    result = []

    with torch.no_grad():

        for f in factors:
            z = im_z + (f * attr_z).type(torch.FloatTensor).to(device)
            im = torch.squeeze(model.decode(z).cpu())
            result.append(im)

    return result


def plot_loss(train_loss, test_loss, filepath):
    train_x, train_l = zip(*train_loss)
    test_x, test_l = zip(*test_loss)
    plt.figure()
    plt.title('Train Loss vs. Test Loss')
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.plot(train_x, train_l, 'b', label='train_loss')
    plt.plot(test_x, test_l, 'r', label='test_loss')
    plt.legend()
    plt.savefig(filepath)


# all images with attribute 
def get_attr_ims(attr, num=5, has=True, gender="male"):
    attr_map, id_attr_map = prep.get_attributes()

    ids = prep.get_attr(attr_map, id_attr_map, attr, has, gender)

    # subset of dataset that contains img with those attributes 
    dataset = prep.ImageDiskLoader(ids)

    # return <num> indices within dataset 
    indices = np.random.randint(0, len(dataset), num)
    ims = [dataset[i] for i in indices]
    idx_ids = [dataset.im_ids[i] for i in indices]

    # returns <num> images with attribute 
    # each returned img is shape(3,64,64)
    # each idx_id: xxxxx.jpg
    return ims, idx_ids

