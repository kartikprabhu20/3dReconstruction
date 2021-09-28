"""

    Created on 22/05/21 6:42 AM
    @author: Kartik Prabhu

"""
import os

import cv2
import torch
import torch.utils.data as data
import math

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold
from torchvision import models, transforms

import Baseconfig
from Datasets.PhotorealisticDataset import PhotoRealDataset
from Datasets.pix3d.Pix3dDataset import Pix3dDataset
from Datasets.pix3dsynthetic.SyntheticPix3dDataset import SyntheticPix3d, SyntheticPix3dDataset
import seaborn as sns


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def plot_images(images, labels, classes, normalize = True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (10, 10))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')

# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        print(imageData[i].shape)
        img = imageData[i].permute(1, 2, 0) *255.
        img = img.numpy()
        img = img.astype(np.uint8)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

def computeTSNEProjectionOfPixelSpace(X,imageSize =224, display=True,zoom= 0.1):
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X.reshape([-1,imageSize*imageSize*3]))

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")

        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=zoom)
        plt.show()
    else:
        return X_tsne

# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace(X, encoder, display=True, zoom= 0.3, transform=None):
    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = encoder(transform(X))

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded.detach().numpy())

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=zoom)
        plt.show()
    else:
        return

# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace_targets(X,Y, encoder,title ="T-SNE", transform=None, classes=7):
    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = encoder(transform(X))
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded.detach().numpy())
    centroid =(sum(X_tsne[:, 0]) / len(X_tsne[:, 0]), sum(X_tsne[:, 1]) / len(X_tsne[:, 1]))
    print(X_tsne[:, 0])
    print(len(X_tsne[:, 0]))
    print(sum(X_tsne[:, 0]) / len(X_tsne[:, 0]))
    print(centroid[0])
    print(centroid[1])

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=Y, legend='full', palette=sns.color_palette("tab20", classes)).set(title=title)
    plt.legend(bbox_to_anchor=(1.11, 1),borderaxespad=0)
    plt.show()

    # Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace_targets_2(X,Y, encoder,title ="T-SNE", transform=None, classes=7, images_per_cat =10, fileName=None):
    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = encoder(transform(X))
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    pallete =["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
          "#DEBB9B", "#FAB0E4","#B8850A", "#006374"][0: classes]

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded.detach().numpy())

    centroids = []
    for i in range(classes):
        centroid =[]
        centroid.append(sum(X_tsne[i*images_per_cat:i*images_per_cat+images_per_cat, 0]) / images_per_cat)
        centroid.append(sum(X_tsne[i*images_per_cat:i*images_per_cat+images_per_cat, 1]) / images_per_cat)
        centroids.append(centroid)

    plt.figure(figsize=(12, 10))

    print(centroids)

    for i in range(classes):
        color = pallete[i]
        for j in range(images_per_cat):
            x = [X_tsne[i*images_per_cat+j, 0], centroids[i][0]]
            y = [X_tsne[i*images_per_cat+j, 1], centroids[i][1]]
            plt.plot(x, y, c=color, alpha=0.8 if i==classes-1 or i==classes-2 else 0.2)

    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=Y, legend='full', palette=pallete).set(title=title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(fileName, bbox_inches='tight',format='pgf')
    # plt.show()

    return centroids

def computeTSNEProjectionOfLatentSpace_targets_3(X,Y, encoder,title ="T-SNE", transform=None, classes=7, images_per_cat =10, fileName=None):
    # Compute latent space representation

    plt.clf()
    print("Computing latent space projection...")
    X_encoded = encoder(transform(X))
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    pallete =["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
              "#DEBB9B", "#FAB0E4","#B8850A", "#006374"][0: classes]

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded.detach().numpy())
    print(X_tsne.shape)
    centroids = []
    for i in range(classes):
        centroid =[]
        centroid.append(sum(X_tsne[i*images_per_cat:i*images_per_cat+images_per_cat, 0]) / images_per_cat)
        centroid.append(sum(X_tsne[i*images_per_cat:i*images_per_cat+images_per_cat, 1]) / images_per_cat)
        centroids.append(centroid)

    plt.figure(figsize=(12, 10))
    # pallete = sns.color_palette("pastel", classes)[0:7]
    # pallete.append(sns.color_palette("dark", classes)[7:9])



    print(centroids)

    for i in range(classes):
        color = pallete[i]
        for j in range(images_per_cat):
            x = [X_tsne[i*images_per_cat+j, 0], centroids[i][0]]
            y = [X_tsne[i*images_per_cat+j, 1], centroids[i][1]]
            plt.plot(x, y, c=color, alpha=0.8 if i==classes-1 or i==classes-2 else 0.2)

    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=Y, legend='full', palette=pallete).set(title=title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fileName, bbox_inches='tight',format='pgf')
    # plt.show()

    return centroids

def plot(images,batch_size,img_size):
    for i in range(len(images)):
    # display original
        ax = plt.subplot(16, 8, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # # display reconstruction
        # ax = plt.subplot(2, batch_size, i + 1 + batch_size)
        # plt.imshow(images[i].reshape(img_size, img_size, 3))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

    plt.show()

def distances(points):
    # points = [[-0.5194042528580342, 4.374773686485631], [-2.382752076190497, -2.5119599529675076], [1.449307121136891, 1.2617859169840813], [-2.6915156548576697, -0.14893598801323346], [-3.8478046054286614, -1.4560019332649452], [-1.7069824699844633, 3.193948324037982], [-1.5260764980422599, -0.2056036852300167]]

    distances_list = []
    for i in range(0, len(points)-1):
        print(i)
        dist = calculate_distance(points[len(points)-1][0],points[len(points)-1][1],points[i][0],points[i][1])
        # print()
        distances_list.append(dist)

    return distances_list

def calculate_distance(x1,y1,x2,y2):
    return math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

if __name__ == '__main__':

    pretrained_model = models.vgg16(pretrained = True)

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

    config = Baseconfig.config

    images_per_category = 50


    config.dataset_path ='/Users/apple/OVGU/Thesis/Dataset/pix3d'
    pix3dtestdataset = Pix3dDataset(config).get_img_trainset(transform1,images_per_category=images_per_category)
    pix3d_test_loader = torch.utils.data.DataLoader(pix3dtestdataset, batch_size=images_per_category * len(pix3dtestdataset.unique_labels()), shuffle=True,
                                           num_workers=0)
    pix3dIter =  iter(pix3d_test_loader)
    # config.dataset_path ='/Users/apple/OVGU/Thesis/s2r3dfree_v1'
    # synthpix3dtestdataset = SyntheticPix3dDataset(config).get_img_trainset(transform1,images_per_category=images_per_category)
    # synth_pix3d_test_loader = torch.utils.data.DataLoader(synthpix3dtestdataset, batch_size=images_per_category * len(synthpix3dtestdataset.unique_labels()), shuffle=True,
    #                                                 num_workers=0)
    # print(synthpix3dtestdataset.__len__())
    print(pix3dtestdataset.__len__())

    # synthIter =  iter(synth_pix3d_test_loader)
    # synth_images, local_labels = next(synthIter)
    # computeTSNEProjectionOfLatentSpace_targets(synth_images,local_labels,pretrained_model,title="S2R:3D-FREE", transform= transform,classes=len(synthpix3dtestdataset.unique_labels()))

    pix3d_images, local_labels2 = next(pix3dIter)
    # computeTSNEProjectionOfLatentSpace_targets(pix3d_images,local_labels2,pretrained_model,title="Pix3D", transform= transform,classes=len(pix3dtestdataset.unique_labels()))

    # images = torch.cat([synth_images, pix3d_images], axis=0)
    # labels = local_labels+ local_labels2
    # computeTSNEProjectionOfLatentSpace_targets(images,labels,pretrained_model,title="S2R:3D-FREE + Pix3D", transform= transform,classes=14)

##############################################################################################################################
#     # # Load in the images
#     root_path ='/Users/apple/OVGU/Thesis/images/survey'
#     images_per_cat=28
#     photoRealistDataset = PhotoRealDataset(root_path,transforms=transform1,images_per_cat=images_per_cat)
#     photorealistic_loader = torch.utils.data.DataLoader(photoRealistDataset, batch_size=photoRealistDataset.__len__(), shuffle=False,
#                                                     num_workers=0)
#     print(photoRealistDataset.__len__())
#     photoIter =  iter(photorealistic_loader)
#     photorealistic_images, labels = next(photoIter)
#     print(labels)
#     uniq_labels = photoRealistDataset.get_unique_labels()
#     print(uniq_labels)
#     centroids = computeTSNEProjectionOfLatentSpace_targets_2(photorealistic_images, labels, pretrained_model, title="Photo-Realistic Images",
#                                                              transform= transform, classes=len(uniq_labels), images_per_cat=images_per_cat,
#                                                              fileName='/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/domaingap/photorealisitic_dataset_2.pgf')
# # ##############################################################################################################################
#     photoRealistDataset1 = PhotoRealDataset('/Users/apple/OVGU/Thesis/images/survey1',transforms=transform1,images_per_cat=images_per_cat)
#     photorealistic_loader1 = torch.utils.data.DataLoader(photoRealistDataset1, batch_size=photoRealistDataset1.__len__(), shuffle=False,
#                                                         num_workers=0)
#     pix3d = PhotoRealDataset('/Users/apple/OVGU/Thesis/images/survey2',transforms=transform1,images_per_cat=images_per_cat)
#     photorealistic_loader2 = torch.utils.data.DataLoader(pix3d, batch_size=pix3d.__len__(), shuffle=False,
#                                                          num_workers=0)
#
#     photos, labels =   next(iter(photorealistic_loader1))
#     pix, lab = next(iter(photorealistic_loader2))
#
#     for i in range(0,int (photoRealistDataset1.__len__()/images_per_cat)):
#         photorealistic_images = photos[i*images_per_cat: i*images_per_cat+images_per_cat]
#
#         print(photorealistic_images.shape)
#         print(pix.shape)
#
#         imgs = torch.cat([photorealistic_images,pix], dim=0)
#         print(imgs.shape)
#
#         local_labels = labels[i*images_per_cat: i*images_per_cat+images_per_cat]
#         all_labels = local_labels + lab
#
#         centroids = computeTSNEProjectionOfLatentSpace_targets_3(imgs, all_labels, pretrained_model, title="Photo-Realistic Images",
#                                                              transform= transform, classes=2, images_per_cat=images_per_cat,
#                                                              fileName='/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/domaingap/'+local_labels[0]+'_Pix3d'+'.pgf')
#
# ##############################################################################################################################
    config = Baseconfig.config
    images_per_cat = 40
    batch_size = 7*images_per_cat


    config.dataset_path ='/Users/apple/OVGU/Thesis/Dataset/pix3d'
    config.pix3d.train_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_train_2.npy'
    config.pix3d.test_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_test_2.npy'

    pix3dtestdataset = Pix3dDataset(config).get_img_testset(transform1,images_per_category=images_per_cat)
    pix3d_test_loader = torch.utils.data.DataLoader(pix3dtestdataset, batch_size=batch_size, shuffle=True,
                                                num_workers=0)

    config.dataset_path ='/Users/apple/OVGU/Thesis/s2r3dfree_v3'
    config.s2r3dfree.train_test_split = config.root_path+"/Datasets/pix3dsynthetic/splits/train_test_split_modelwise_v3.p"

    synthtestdataset = SyntheticPix3dDataset(config).get_img_testset(transform1,images_per_category=images_per_cat)
    s2r_test_loader = torch.utils.data.DataLoader(synthtestdataset, batch_size=batch_size, shuffle=True,
                                                num_workers=0)


    pix3d, pix3d_labels = next(iter(pix3d_test_loader))
    s2r, s2r_labels = next(iter(s2r_test_loader))

    print(pix3d.shape)
    print(s2r.shape)

    imgs = torch.cat([pix3d,s2r], dim=0)
    print(imgs.shape)

    all_labels = ['Pix3D'] * len(pix3d_labels) + ['S2R:3DFREE'] * len(s2r_labels)

    centroids = computeTSNEProjectionOfLatentSpace_targets_3(imgs, all_labels, pretrained_model, title="T-SNE for Pix3D and S2R:3DFREE",
                                                             transform= transform, classes=2, images_per_cat=batch_size,
                                                             fileName='/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/domaingap/s2r3dfree_Pix3d'+str(batch_size)+'.pgf')

# ##############################################################################################################################
    config = Baseconfig.config
    images_per_cat = 50
    batch_size = images_per_cat

    config.dataset_path ='/Users/apple/OVGU/Thesis/Dataset/pix3d'
    config.pix3d.train_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_train_chair.npy'
    config.pix3d.test_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_test_chair.npy'

    pix3dtestdataset = Pix3dDataset(config).get_img_testset(transform1,images_per_category=images_per_cat)
    pix3d_test_loader = torch.utils.data.DataLoader(pix3dtestdataset, batch_size=batch_size, shuffle=True,
                                                num_workers=0)
    pix3d, pix3d_labels = next(iter(pix3d_test_loader))
    config.s2r3dfree.train_test_split = config.root_path+"/Datasets/pix3dsynthetic/splits/train_test_split_modelwise_chair_10000.p"

    datasets = ['s2r3dfree_textureless','s2r3dfree_textureless_light','s2r3dfree_background','s2r3dfree_background_light2','s2r3dfree_chair']
    for data in datasets:

        config.dataset_path ='/Users/apple/OVGU/Thesis/'+data

        synthtestdataset = SyntheticPix3dDataset(config).get_img_testset(transform1,images_per_category=images_per_cat)
        s2r_test_loader = torch.utils.data.DataLoader(synthtestdataset, batch_size=batch_size, shuffle=True,
                                              num_workers=0)


        s2r, s2r_labels = next(iter(s2r_test_loader))

        print(pix3d.shape)
        print(s2r.shape)

        imgs = torch.cat([pix3d,s2r], dim=0)
        print(imgs.shape)

        all_labels = ['Pix3D'] * len(pix3d_labels) + [data] * len(s2r_labels)

        centroids = computeTSNEProjectionOfLatentSpace_targets_3(imgs, all_labels, pretrained_model, title="T-SNE for chair images "+"("+data+")",
                                                         transform= transform, classes=2, images_per_cat=images_per_category,
                                                         fileName='/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/domaingap/Pix3d_'+data+'.pgf')

###############################################################################################################################

    config = Baseconfig.config
    images_per_cat = 50
    batch_size = images_per_cat

    config.dataset_path ='/Users/apple/OVGU/Thesis/Dataset/pix3d'
    config.pix3d.train_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_train_chair.npy'
    config.pix3d.test_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_test_chair.npy'

    pix3dtestdataset = Pix3dDataset(config).get_img_testset(transform1,images_per_category=images_per_cat)
    pix3d_test_loader = torch.utils.data.DataLoader(pix3dtestdataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=0)
    pix3d, pix3d_labels = next(iter(pix3d_test_loader))
    config.s2r3dfree.train_test_split = config.root_path+"/Datasets/pix3dsynthetic/splits/train_test_split_modelwise_chair_10000.p"

    datasets = ['s2r3dfree_textureless','s2r3dfree_textureless_light','s2r3dfree_background','s2r3dfree_background_light2','s2r3dfree_chair']

    all_labels = ['Pix3D'] * len(pix3d_labels)
    imgs = pix3d

    for data in datasets:

        config.dataset_path ='/Users/apple/OVGU/Thesis/'+data

        synthtestdataset = SyntheticPix3dDataset(config).get_img_testset(transform1,images_per_category=images_per_cat)
        s2r_test_loader = torch.utils.data.DataLoader(synthtestdataset, batch_size=batch_size, shuffle=True,
                                                      num_workers=0)

        s2r, s2r_labels = next(iter(s2r_test_loader))
        imgs = torch.cat([imgs,s2r], dim=0)
        all_labels = all_labels + [data] * len(s2r_labels)

    centroids = computeTSNEProjectionOfLatentSpace_targets_3(imgs, all_labels, pretrained_model, title="T-SNE for chair images with domain randomisation",
                                                                 transform= transform, classes=len(datasets)+1, images_per_cat=images_per_category,
                                                                 fileName='/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/evaluation/domaingap/Pix3d_chair_DR.pgf')

###############################################################################################################################
