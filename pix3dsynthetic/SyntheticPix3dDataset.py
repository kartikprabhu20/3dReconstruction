"""

    Created on 23/04/21 10:13 AM
    @author: Kartik Prabhu

"""

import os

import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import pickle
import cv2
from PIL import Image
import config

from random import seed
from random import randint

from pix3dsynthetic.DataExploration import get_train_test_split
from utils import load_mat_pix3d, mat_to_array, load

torch.manual_seed(2020)
np.random.seed(2020)
seed(2020)


class SyntheticPix3d(Dataset):
    def __init__(self,config,input_paths,output_paths):
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.config = config

        #paths
        self.root_path = config.root_path
        self.root_img_path = os.path.join(self.root_path,"imgs")
        self.root_depthmap_path = os.path.join(self.root_path ,"depthmaps")
        self.root_masks_path = os.path.join(self.root_path,"masks")
        self.root_models_path = os.path.join(self.root_path,"models")

        self.resize = config.resize
        self.size = config.size
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_img_path,self.input_paths[idx])
        output_model_path = os.path.join(os.path.join(self.root_models_path,self.output_paths[idx]),
                                         "model.obj" if self.config.is_mesh else "voxel.mat")

        input_img = self.transform(self.read_img(img_path))
        input_stack = input_img

        if self.config.include_depthmaps:
            depthmap_path = os.path.join(self.root_depthmap_path,self.input_paths[idx])
            input_depth = self.transform(self.read_img(depthmap_path))
            input_stack = torch.cat((input_stack, input_depth), 0)

        if self.config.include_masks:
            mask_path = os.path.join(self.root_masks_path,self.input_paths[idx])
            input_mask = self.transform(self.read_img(mask_path))
            input_stack = torch.cat((input_stack, input_mask), 0)

        # if self.config.include_edges:

        print(input_stack.shape)

        if self.config.is_mesh:
            output_model = load(output_model_path)
        else:
            output_model = torch.tensor(mat_to_array(output_model_path))
            print(output_model.shape)

        return input_img, output_model

    def read_img(self, path):
        with Image.open(path) as img:
            img = img.convert('RGB')# convert('L') if it's a gray scale image
            if self.resize:
                img = img.resize((self.size[0],self.size[1]), Image.ANTIALIAS)
        return img

    def __len__(self):
        return len(self.input_paths)

if __name__ == '__main__':
    config = config.config

    train_img_list,train_model_list,test_img_list,test_model_list = get_train_test_split("train_test_split.p")
    traindataset = SyntheticPix3d(config,train_img_list,train_model_list)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=8)

    for batch_index, (input_batch, input_label) in enumerate(train_loader):
        # input_batch, input_label = input_batch[:, None, :].cuda(), input_label[:, None, :].cuda()
        print(input_batch.shape)
        print(input_label.shape)

        break



