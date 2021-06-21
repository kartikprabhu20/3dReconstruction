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
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded.detach().numpy())

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=zoom)
        plt.show()
    else:
        return

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


    # IN_FEATURES = pretrained_model.classifier[-1].in_features
    OUTPUT_DIM = 10
    # final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    # pretrained_model.classifier[-1] = final_fc
    #
    # for parameter in pretrained_model.features.parameters():
    #     parameter.requires_grad = False
    #
    # for parameter in pretrained_model.classifier[:-1].parameters():
    #     parameter.requires_grad = False
    #
    # pretrained_size = 224
    # pretrained_means = [0.485, 0.456, 0.406]
    # pretrained_stds= [0.229, 0.224, 0.225]
    #
    # custom_transforms = transforms.Compose([
    #     transforms.Resize(pretrained_size),
    #     transforms.RandomRotation(5),
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.RandomCrop(pretrained_size, padding = 10),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean = pretrained_means,
    #                          std = pretrained_stds)
    # ])
    #
    # final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    # pretrained_model.classifier[-1] = final_fc

    # imageSize = 512
    #
    # config = Baseconfig.config
    # print("configuration: " + str(config))
    # MODEL_NAME = config.main_name
    # ROOT_PATH = config.root_path
    #
    # train_img_list,train_model_list,test_img_list,test_model_list = get_train_test_split(ROOT_PATH+"/pix3dsynthetic/train_test_split.p")
    #
    # testdataset = SyntheticPix3d(config,test_img_list,test_model_list)
    # dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=128)
    # test_images, models = next(iter(dataloader))
    # print(test_images.shape)
    # # plt.imshow(test_images[1].permute(1, 2, 0))
    # # plt.show()
    # # plot(test_images,batch_size=128,img_size=512)
    # print("t-SNE projection of image space representations from the validation set")
    # computeTSNEProjectionOfPixelSpace(test_images, imageSize= test_images[0].shape[1])
    # print("t-SNE projection of latent space representations from the validation set")
    # # computeTSNEProjectionOfLatentSpace(test_images[:250], encoder)

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

    config.dataset_path ='/Users/apple/OVGU/Thesis/Dataset/pix3d'
    pix3dtestdataset = Pix3dDataset(config).get_img_trainset(transform1)
    pix3d_test_loader = torch.utils.data.DataLoader(pix3dtestdataset, batch_size=100, shuffle=True,
                                           num_workers=0)
    pix3dIter =  iter(pix3d_test_loader)

    config.dataset_path ='/Users/apple/OVGU/Thesis/SynthDataset1'
    synthpix3dtestdataset = SyntheticPix3dDataset(config).get_img_testset(transform1)
    synth_pix3d_test_loader = torch.utils.data.DataLoader(synthpix3dtestdataset, batch_size=100, shuffle=True,
                                                    num_workers=0)
    synthIter =  iter(synth_pix3d_test_loader)
    synth_images, local_labels = next(synthIter)
    computeTSNEProjectionOfLatentSpace(synth_images,pretrained_model, transform= transform)

    pix3d_images, local_labels2 = next(pix3dIter)
    computeTSNEProjectionOfLatentSpace(pix3d_images,pretrained_model, transform= transform)

    images = torch.cat([synth_images, pix3d_images], axis=0)
    print(images.shape)
    computeTSNEProjectionOfLatentSpace(images,pretrained_model, zoom=0.2, transform= transform)

    # embedding = []
    # for batch_index, (local_batch, local_labels) in enumerate(test_loader):
    #     print(local_batch.shape)
    #     input = torch.unsqueeze(local_batch, 0)
    #     embedd = pretrained_model(local_batch)
    #     # print(embedd)
    #     embedding.append(embedd)
    #     computeTSNEProjectionOfLatentSpace(local_batch,pretrained_model)
    #     break
    # print("embedding")
    # print(embedding)
