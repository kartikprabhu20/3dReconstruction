from sklearn.feature_extraction import image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json


import numpy as np
import pandas as pd



if __name__ == '__main__':

    # pix3d_json_path = '/Users/apple/OVGU/Thesis/Dataset/pix3d/pix3d.json'
    # y = json.load(open(pix3d_json_path))
    # print(y[1])
    img = np.load('/Users/apple/Downloads/single.val/0a2/0a2a7e957087384f1163964b4507846e8fc4f426df44e77043567db7583e7d73.npz')
    img = img.f.arr_0
    print(img.shape)
    plt.ion()
    plt.figure()
    plt.imshow(img)


