"""

    Created on 22/05/21 6:42 AM
    @author: Kartik Prabhu

"""
import torch
import torch.utils.data as data

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold
from torchvision import models, transforms

import Baseconfig
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
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=Y, legend='full', palette=sns.color_palette("tab20", classes)).set(title=title)
    plt.legend(bbox_to_anchor=(1.11, 1),borderaxespad=0)
    plt.show()

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
    pix3dtestdataset = Pix3dDataset(config).get_img_testset(transform1,images_per_category=images_per_category)
    pix3d_test_loader = torch.utils.data.DataLoader(pix3dtestdataset, batch_size=images_per_category * len(pix3dtestdataset.unique_labels()), shuffle=True,
                                           num_workers=0)
    pix3dIter =  iter(pix3d_test_loader)

    config.dataset_path ='/Users/apple/OVGU/Thesis/SynthDataset1'
    synthpix3dtestdataset = SyntheticPix3dDataset(config).get_img_testset(transform1,imagesPerCategory=images_per_category)
    synth_pix3d_test_loader = torch.utils.data.DataLoader(synthpix3dtestdataset, batch_size=images_per_category * len(pix3dtestdataset.unique_labels()), shuffle=True,
                                                    num_workers=0)
    print(synthpix3dtestdataset.__len__())
    print(pix3dtestdataset.__len__())

    synthIter =  iter(synth_pix3d_test_loader)
    synth_images, local_labels = next(synthIter)
    computeTSNEProjectionOfLatentSpace_targets(synth_images,local_labels,pretrained_model,title="S2R:3D-FREE", transform= transform)

    pix3d_images, local_labels2 = next(pix3dIter)
    computeTSNEProjectionOfLatentSpace_targets(pix3d_images,local_labels2,pretrained_model,title="Pix3D", transform= transform)

    images = torch.cat([synth_images, pix3d_images], axis=0)
    labels = local_labels+ local_labels2
    computeTSNEProjectionOfLatentSpace_targets(images,labels,pretrained_model,title="S2R:3D-FREE + Pix3D", transform= transform,classes=14)

