import cv2
import torch
from sklearn.feature_extraction import image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json


import numpy as np
import pandas as pd
import os


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)

def get_classes(root_path):
    #return ["bed","bookcase","chair","desk","sofa","table","wardrobe"]
    return get_classes_path(root_path)

def get_classes_path(path):
    #return ["bed","bookcase","chair","desk","sofa","table","wardrobe"]
    return [class_folder for class_folder in os.listdir(path) if not class_folder.startswith('.')]


if __name__ == '__main__':

    # pix3d_json_path = '/Users/apple/OVGU/Thesis/Dataset/pix3d/pix3d.json'
    # y = json.load(open(pix3d_json_path))
    # print(y[1])
    # img = np.load('/Users/apple/Downloads/single.val/0a2/0a2a7e957087384f1163964b4507846e8fc4f426df44e77043567db7583e7d73.npz')
    # img = img.f.arr_0
    # print(img.shape)
    # plt.ion()
    # plt.figure()
    # plt.imshow(img)


    root_path = '/Users/apple/OVGU/Thesis/s2r3dfree_chair_light'

    total = 0
    for label in get_classes(root_path):
        print(label)

        label_length=0
        labelPath = os.path.join(root_path,label)
        for folder in os.listdir(labelPath):

            if(folder == ".DS_Store"):
                continue
            label_length += len(os.listdir(os.path.join(labelPath,folder)))

        print(label_length)
        total += label_length


    print(total)

    #
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.stats import multivariate_normal
    # from scipy.stats import dirichlet
    # from torch.distributions.multivariate_normal import MultivariateNormal
    #
    #
    # m = 20 # sample size
    # x_mean = torch.zeros(2)+1
    # y_mean = torch.zeros(2)
    # x_cov = 2*torch.eye(2) # IMPORTANT: Covariance matrices must be positive definite
    # y_cov = 3*torch.eye(2) - 1
    #
    # px = MultivariateNormal(x_mean, x_cov)
    # qy = MultivariateNormal(y_mean, y_cov)
    # x = px.sample([m]).to(device)
    # y = qy.sample([m]).to(device)
    #
    # print(x.shape)
    # print(y.shape)
    #
    # result = MMD(x, y, kernel="multiscale")
    #
    # print(f"MMD result of X and Y is {result.item()}")
    #
    # # ---- Plotting setup ----
    #
    # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4), dpi=100)
    # #plt.tight_layout()
    # delta = 0.025
    #
    # x1_val = np.linspace(-5, 5, num=m)
    # x2_val = np.linspace(-5, 5, num=m)
    #
    # x1, x2 = np.meshgrid(x1_val, x2_val)
    #
    # px_grid = torch.zeros(m,m)
    # qy_grid = torch.zeros(m,m)
    #
    #
    # for i in range(m):
    #     for j in range(m):
    #         px_grid[i,j] = multivariate_normal.pdf([x1_val[i],x2_val[j]], x_mean, x_cov)
    #         qy_grid[i,j] = multivariate_normal.pdf([x1_val[i],x2_val[j]], y_mean, y_cov)
    #
    #
    # CS1 = ax1.contourf(x1, x2, px_grid,100, cmap=plt.cm.YlGnBu)
    # ax1.set_title("Distribution of $X \sim P(X)$")
    # ax1.set_ylabel('$x_2$')
    # ax1.set_xlabel('$x_1$')
    # ax1.set_aspect('equal')
    # ax1.scatter(x[:10,0].cpu(), x[:10,1].cpu(), label="$X$ Samples", marker="o", facecolor="r", edgecolor="k")
    # ax1.legend()
    #
    # CS2 = ax2.contourf(x1, x2, qy_grid,100, cmap=plt.cm.YlGnBu)
    # ax2.set_title("Distribution of $Y \sim Q(Y)$")
    # ax2.set_xlabel('$y_1$')
    # ax2.set_ylabel('$y_2$')
    # ax2.set_aspect('equal')
    # ax2.scatter(y[:10,0].cpu(), y[:10,1].cpu(), label="$Y$ Samples", marker="o", facecolor="r", edgecolor="k")
    # ax2.legend()
    # #ax1.axis([-2.5, 2.5, -2.5, 2.5])
    #
    # # Add colorbar and title
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    # cbar = fig.colorbar(CS2, cax=cbar_ax)
    # cbar.ax.set_ylabel('Density results')
    # plt.suptitle(f"MMD result: {round(result.item(),3)}",y=0.95, fontweight="bold")
    # plt.show()




