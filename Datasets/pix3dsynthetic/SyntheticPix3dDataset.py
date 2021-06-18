"""

    Created on 23/04/21 10:13 AM
    @author: Kartik Prabhu

"""

import os
import pickle

import cv2
import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import Baseconfig

from Datasets.pix3dsynthetic.BaseDataset import BaseDataset
from utils import mat_to_array, load

from random import seed
torch.manual_seed(2020)
np.random.seed(2020)
seed(2020)

class SyntheticPix3dDataset(BaseDataset):
    def __init__(self,config):
        print("SyntheticPix3dDataset")
        self.config=config
        self.train_img_list,self.train_model_list,self.test_img_list,self.test_model_list = self.get_train_test_split(self.config.root_path+"/Datasets/pix3dsynthetic/train_test_split.p")

    def get_trainset(self, transforms=None):
        return SyntheticPix3d(self.config,self.train_img_list,self.train_model_list, transforms=transforms)

    def get_testset(self, transforms=None):
        return SyntheticPix3d(self.config,self.test_img_list,self.test_model_list, transforms=transforms)

    def get_train_test_split(self,filePath):
        pickle_file = pickle.load(open(filePath, "rb" ))
        x = pickle_file['train_x']
        y = pickle_file['train_y']
        xt = pickle_file['test_x']
        yt = pickle_file['test_y']

        return x,y,xt,yt

class SyntheticPix3d(Dataset):
    def __init__(self,config,input_paths,output_paths, transforms=None):
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.config = config

        #paths
        self.dataset_path = config.dataset_path
        self.dataset_img_path = os.path.join(self.dataset_path,"imgs")
        self.dataset_depthmap_path = os.path.join(self.dataset_path ,"depthmaps")
        self.dataset_masks_path = os.path.join(self.dataset_path,"masks")
        self.dataset_models_path = os.path.join(self.dataset_path,"models")

        self.resize = config.resize
        self.size = config.size
        self.transform = transforms


    def __getitem__(self, idx):

        img_path = os.path.join(self.dataset_img_path,self.input_paths[idx])
        taxonomy_id =  self.input_paths[idx].split('/')[0] # 'chair/IKEA_EKENAS/11.png' = chair

        if self.config.is_mesh:
            output_model_path = os.path.join(os.path.join(self.dataset_models_path,self.output_paths[idx]),
                                         "model.obj")
        else:
            output_model_path = os.path.join(os.path.join(self.dataset_models_path,self.output_paths[idx]),
                                 "voxel_"+str(self.config.voxel_size)+".npy" if self.config.label_type == Baseconfig.LabelType.NPY else "voxel.mat")
        # print(str(idx)+":img_path:"+img_path)
        # print(str(idx)+":output_model_path:"+output_model_path)
        input_img = self.read_img(img_path)
        input_stack = input_img

        if self.config.include_depthmaps:
            depthmap_path = os.path.join(self.dataset_depthmap_path,self.input_paths[idx])
            input_depth = self.read_img(depthmap_path,type='L')
            input_stack = np.vstack((input_stack, input_depth))

        if self.config.include_masks:
            mask_path = os.path.join(self.dataset_masks_path,self.input_paths[idx])
            input_mask = self.read_img(mask_path,type='L')
            input_stack = np.vstack((input_stack, input_mask))


        input_stack = self.transform(input_stack)
        # input_stack =input_stack[None, :]
        # if self.config.include_edges:

        # print(input_stack.shape)

        if self.config.is_mesh:
            output_model = load(output_model_path)
        else:
            if self.config.label_type == Baseconfig.LabelType.NPY:
                output_model = torch.tensor(np.load(output_model_path), dtype=torch.float32)
            else:
                output_model = torch.tensor(mat_to_array(output_model_path), dtype=torch.float32)

        # print(output_model.shape)

        return taxonomy_id,input_stack, output_model

    def read_img(self, path, type='RGB'):
        # with Image.open(path) as img:
        #     img = img.convert(type)# convert('L') if it's a gray scale image
        #     if self.resize:
        #         img = img.resize((self.size[0],self.size[1]), Image.ANTIALIAS)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if self.resize:
            img = cv2.resize(img, (self.size[0],self.size[1]), interpolation = cv2.INTER_AREA)

        if len(img.shape) > 1:
            img = img / 255.
        if len(img.shape) < 3:
            print('[WARN] It seems the image file %s is grayscale.' % (path))
            img = np.stack((img, ) * 3, -1)

        img = np.asarray([img])
        return img

    def __len__(self):
        # if self.config.platform == "darwin":
        #     return 8 #debug
        return len(self.input_paths)

if __name__ == '__main__':
    config = Baseconfig.config

    train_img_list,train_model_list,test_img_list,test_model_list = get_train_test_split(
        "/Datasets/pix3dsynthetic/train_test_split.p")
    traindataset = SyntheticPix3d(config,train_img_list,train_model_list)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=8)

    for batch_index, (input_batch, input_label) in enumerate(train_loader):
        # input_batch, input_label = input_batch[:, None, :].cuda(), input_label[:, None, :].cuda()
        print(input_batch.shape)
        print(input_label.shape)

        print(input_batch[0][0:3].shape)
        break



