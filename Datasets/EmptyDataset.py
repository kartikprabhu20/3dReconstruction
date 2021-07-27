"""

    Created on 20/07/21 11:35 AM 
    @author: Kartik Prabhu

"""
import cv2
import torch
from PIL import Image
import os
from torch.utils.data import Dataset
from Datasets.pix3dsynthetic.BaseDataset import BaseDataset
import numpy as np


class EmptyDataset(BaseDataset):
    def __init__(self,config):
        self.config=config

    def get_img_testset(self, transforms=None,images_per_category=0):
        return EmptyData(self.config,customtransforms=transforms)

class EmptyData(Dataset):
    def __init__(self, config, customtransforms=None):
        data_path = config.empty_data_path
        self.image_paths = []
        for image_path in os.listdir(data_path):
            input_path = os.path.join(data_path, image_path)
            self.image_paths.append(input_path)

        self.resize = config.resize
        self.size = config.size
        self.transform = customtransforms

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        input_img = self.read_img(img_path)
        input_stack = self.transform(input_img)

        #dummy, not needed
        taxonomy_id =0
        output_model = ""

        return input_stack

    def read_img(self, path, type='RGB'):
        # with Image.open(path) as img:
        #     img = img.convert(type)# convert('L') if it's a gray scale image
        #     if self.resize:
        #         img = img.resize((self.size[0],self.size[1]), Image.ANTIALIAS)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if self.resize:
            img = cv2.resize(img, (self.size[0],self.size[1]), interpolation = cv2.INTER_AREA)/ 255.


        if len(img.shape) < 3:
            print('[WARN] It seems the image file %s is grayscale.' % (path))
            img = np.stack((img, ) * 3, -1)
        elif img.shape[2] > 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img = np.asarray([img])
        return img

    def __len__(self):
        # return 10
        return len(self.image_paths)

