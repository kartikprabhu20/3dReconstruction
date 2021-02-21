"""

    Created on 21/02/21 5:26 PM 
    @author: Kartik Prabhu

"""
import skimage.io as io
import numpy as np
import h5py
from PIL import Image
import os
import torch
try:
    import accimage
except ImportError:
    accimage = None
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

def colored_depthmap(depth, d_min=None, d_max=None):
    # cmap = plt.cm.viridis
    cmap = plt.get_cmap('jet')
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C



if __name__ == '__main__':
    # depth = np.clip(np.load("/Users/apple/OVGU/Thesis/images/synthetic/test_depth.png"), 1.0, 10.0) / 10 * 255
    depth = Image.open("/Users/apple/OVGU/Thesis/images/synthetic/test_depth.png").convert('L')
    # depth = Image.fromarray(depth.astype(np.uint8)[:,:], mode='L')
    depth.show()

    depth = np.asarray(depth)
    print(depth.shape)

    rgba_img = colored_depthmap(depth)
    print(rgba_img.shape)
    img_depth = Image.fromarray(rgba_img.astype(np.uint8))
    img_depth.show()