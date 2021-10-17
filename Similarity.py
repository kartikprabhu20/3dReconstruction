"""

    Created on 23/09/21 9:19 PM 
    @author: Kartik Prabhu

"""

import cv2
import torch
import torch.utils.data as data
import math
from torch.distributions import MultivariateNormal
import seaborn as sns  # This is for visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold
from torchvision import models, transforms

import Baseconfig
from Datasets.PhotorealisticDataset import PhotoRealDataset, PhotoRealDataset2
from Datasets.pix3d.Pix3dDataset import Pix3dDataset
from Datasets.pix3dsynthetic.SyntheticPix3dDataset import SyntheticPix3d, SyntheticPix3dDataset
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from torchvision.models import inception_v3

import scipy


def matrix_sqrt(x):
    '''
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a matrix
    '''
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    '''
    Function for returning the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features)
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    '''
    return torch.norm(mu_x - mu_y) + torch.trace(sigma_x + sigma_y - 2 * matrix_sqrt(sigma_x @ sigma_y))


def preprocess(img):
    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    return img


def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))


if __name__ == '__main__':

    root_path = '/Users/apple/OVGU/Thesis/images/survey'

    transform1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    images_per_cat = 30
    photoRealistDataset = PhotoRealDataset2(root_path, transforms=transform1, images_per_cat=images_per_cat)
    photorealistic_loader = torch.utils.data.DataLoader(photoRealistDataset, batch_size=images_per_cat, shuffle=False,
                                                        num_workers=0)

    mse = torch.nn.MSELoss()  # We use Mean squared loss which computes difference between two images.
    kldiv = torch.nn.KLDivLoss(reduction='batchmean')

    print(photorealistic_loader.__len__())

    encoder = models.vgg16(pretrained=True)

    mse_losses = []
    klds = []
    ssim_losses = []
    datasets = []

    input = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 3)
    print(input.shape)
    print(input.shape)
    print(input[0])
    output = kldiv(input, target)
    print(output)

    dataset, input_batch, pix3d = next(iter(photorealistic_loader))
    X_encoded = encoder(transform(transform(input_batch)))
    Y_encoded = encoder(transform(pix3d))
    X_encoded = torch.nn.functional.log_softmax(X_encoded, dim=1)
    Y_encoded = torch.nn.functional.log_softmax(Y_encoded, dim=1)

    print(X_encoded[0])
    print(X_encoded.shape)
    print(Y_encoded.shape)
    kld = kldiv(X_encoded, Y_encoded)
    print(kld)

    for batch_index, (dataset, input_batch, pix3d) in enumerate(photorealistic_loader):
        print(batch_index)

        X_encoded = encoder(transform(input_batch))
        Y_encoded = encoder(transform(pix3d))

        loss_val = mse(X_encoded, Y_encoded)
        kld = kldiv(X_encoded, Y_encoded)
        # ssim_val = ssim(X_encoded, Y_encoded,data_range=Y_encoded.max() - Y_encoded.min())
        # ssim_losses.append(ssim_val)
        mse_losses.append(loss_val)
        klds.append(kld)
        datasets.append(dataset[0])

    print(datasets)
    print(mse_losses)
    print(ssim_losses)
    print(klds)

    inception_model = inception_v3(pretrained=False)
    inception_model.load_state_dict(torch.load("/Users/apple/OVGU/Thesis/inception_v3_google-1a9a5a14.pth"))
    inception_model = inception_model.eval()
    inception_model.fc = torch.nn.Identity()

    photoRealistDataset = PhotoRealDataset2(root_path, transforms=transform1, images_per_cat=images_per_cat)
    photorealistic_loader = torch.utils.data.DataLoader(photoRealistDataset, batch_size=images_per_cat, shuffle=False,
                                                        num_workers=0)

    with torch.no_grad():
        try:
            for batch_index, (dataset, input_batch, pix3d) in enumerate(photorealistic_loader):
                print(dataset[0])
                fake_features_list = []
                real_features_list = []

                real_samples = pix3d
                real_samples = preprocess(real_samples)
                real_features = inception_model(real_samples)
                real_features_list.append(real_features)

                fake_samples = input_batch
                fake_samples = preprocess(fake_samples)
                fake_features = inception_model(fake_samples)
                fake_features_list.append(fake_features)

                fake_features_all = torch.cat(fake_features_list)
                real_features_all = torch.cat(real_features_list)

                mu_fake = torch.mean(fake_features_all, axis=0)
                mu_real = torch.mean(real_features_all, axis=0)
                sigma_fake = get_covariance(fake_features_all)
                sigma_real = get_covariance(real_features_all)

                indices = [2, 4, 5]
                fake_dist = MultivariateNormal(mu_fake[indices], sigma_fake[indices][:, indices])
                fake_samples = fake_dist.sample((5000,))
                real_dist = MultivariateNormal(mu_real[indices], sigma_real[indices][:, indices])
                real_samples = real_dist.sample((5000,))

                df_fake = pd.DataFrame(fake_samples.numpy(), columns=indices)
                df_real = pd.DataFrame(real_samples.numpy(), columns=indices)
                df_fake["is_real"] = "no"
                df_real["is_real"] = "yes"
                df = pd.concat([df_fake, df_real])
                sns.pairplot(df, plot_kws={'alpha': 0.1}, hue='is_real')

                with torch.no_grad():
                    print(frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item())
        except:
            print("Error in loop")

    # ##############################################################################################################################
    config = Baseconfig.config
    images_per_cat = 100
    batch_size = images_per_cat

    config.dataset_path = '/Users/apple/OVGU/Thesis/Dataset/pix3d' if config.platform == "darwin" else '/nfs1/kprabhu/pix3d/'
    config.pix3d.train_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_train_chair.npy'
    config.pix3d.test_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_test_chair.npy'

    pix3dtestdataset = Pix3dDataset(config).get_img_testset(transform1, images_per_category=images_per_cat)
    pix3d_test_loader = torch.utils.data.DataLoader(pix3dtestdataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=0)
    pix3d, pix3d_labels = next(iter(pix3d_test_loader))
    if config.platform != "darwin":
        pix3d = pix3d.cuda()
    config.s2r3dfree.train_test_split = config.root_path + "/Datasets/pix3dsynthetic/splits/train_test_split_modelwise_chair_10000.p"

    datasets = ['s2r3dfree_textureless', 's2r3dfree_textureless_light', 's2r3dfree_background',
                's2r3dfree_background_light2', 's2r3dfree_chair']
    for data in datasets:

        config.dataset_path = '/Users/apple/OVGU/Thesis/' + data if config.platform == "darwin" else '/nfs1/kprabhu/' + data

        synthtestdataset = SyntheticPix3dDataset(config).get_img_testset(transform1, images_per_category=images_per_cat)
        s2r_test_loader = torch.utils.data.DataLoader(synthtestdataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=0)

        s2r, s2r_labels = next(iter(s2r_test_loader))
        if config.platform != "darwin":
            s2r = s2r.cuda()
        print(pix3d.shape)
        print(s2r.shape)

        try:
            print(dataset[0])
            fake_features_list = []
            real_features_list = []

            real_samples = pix3d
            real_samples = preprocess(real_samples)
            real_features = inception_model(real_samples)
            real_features_list.append(real_features)

            fake_samples = s2r
            fake_samples = preprocess(fake_samples)
            fake_features = inception_model(fake_samples)
            fake_features_list.append(fake_features)

            fake_features_all = torch.cat(fake_features_list)
            real_features_all = torch.cat(real_features_list)

            mu_fake = torch.mean(fake_features_all, axis=0)
            mu_real = torch.mean(real_features_all, axis=0)
            sigma_fake = get_covariance(fake_features_all)
            sigma_real = get_covariance(real_features_all)

            indices = [2, 4, 5]
            fake_dist = MultivariateNormal(mu_fake[indices], sigma_fake[indices][:, indices])
            fake_samples = fake_dist.sample((5000,))
            real_dist = MultivariateNormal(mu_real[indices], sigma_real[indices][:, indices])
            real_samples = real_dist.sample((5000,))

            df_fake = pd.DataFrame(fake_samples.numpy(), columns=indices)
            df_real = pd.DataFrame(real_samples.numpy(), columns=indices)
            df_fake["is_real"] = "no"
            df_real["is_real"] = "yes"
            df = pd.concat([df_fake, df_real])
            sns.pairplot(df, plot_kws={'alpha': 0.1}, hue='is_real')

            with torch.no_grad():
                print(data)
                print(frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item())
        except:
            print("Error in loop")

###############################################################################################################################
