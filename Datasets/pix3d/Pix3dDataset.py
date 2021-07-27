"""

    Created on 23/04/21 10:13 AM
    @author: Kartik Prabhu

"""
import json
import os

import cv2
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms

import Baseconfig
import transform_utils

from Datasets.pix3dsynthetic.BaseDataset import BaseDataset
from utils import mat_to_array, load
from sklearn.utils import resample

from random import seed
torch.manual_seed(2020)
np.random.seed(2020)
seed(2020)

class Pix3dDataset(BaseDataset):
    def __init__(self,config):
        self.config=config

        if self.config.pix3d.train_indices.endswith('.npy'):
            self.train_img_list,self.train_model_list,self.test_img_list,\
            self.test_model_list,self.train_category_list,self.test_category_list\
                = self.get_train_test_split_npy(self.config.dataset_path + "/pix3d.json", self.config.pix3d.train_indices,
                                                self.config.pix3d.test_indices,upsample=self.config.upsample)
        else:
            self.train_img_list,self.train_model_list,self.test_img_list, \
            self.test_model_list,self.train_category_list,self.test_category_list \
                = self.get_train_test_split_json(self.config.pix3d.train_indices,self.config.pix3d.test_indices, upsample=self.config.upsample)



    def get_trainset(self, transforms=None,images_per_category=0):
        return Pix3d(self.config,self.train_img_list,self.train_model_list, transforms=transforms,imagesPerCategory=images_per_category)

    def get_testset(self, transforms=None,images_per_category=0):
        return Pix3d(self.config,self.test_img_list,self.test_model_list, transforms=transforms,imagesPerCategory=images_per_category)

    def get_img_trainset(self, transforms=None,images_per_category=0):
        return Pix3d_Img(self.config,self.train_img_list,self.train_category_list,customtransforms=transforms,imagesPerCategory=images_per_category)

    def get_img_testset(self,transforms=None,images_per_category=0):
        return Pix3d_Img(self.config,self.test_img_list,self.test_category_list,customtransforms=transforms,imagesPerCategory=images_per_category)

    def get_train_test_split_npy(self, filePath, train_indices_path, test_indices_path, upsample=False):
        """
        :param filePath: pix3d.json path in dataset folder
        :return:
        """
        y = json.load(open(filePath))
        train, test = self.get_train_test_indices(train_indices_path,test_indices_path)
        # # print(y)
        # print(train)
        # print(test)

        categories = []
        for i in train:
            if not(y[i]['category']=='misc' or y[i]['category']=='tool'):
                categories.append(y[i]['category'])

        final_train_img_list = []
        final_train_model_list = []
        final_train_category_list = []

        for category in set(categories):
            train_img_list = []
            train_model_list= []
            train_category_list = []

            for i in train:
                if y[i]['category']==category:
                    train_img_list.append(y[i]['img'])
                    train_model_list.append(y[i]['model' if self.config.is_mesh else 'voxel'])
                    train_category_list.append(y[i]['category']+'_pix3d')

            final_train_img_list.append(train_img_list)
            final_train_model_list.append(train_model_list)
            final_train_category_list.append(train_category_list)

        train_img_list,train_model_list,train_category_list = self.upsampling(final_train_img_list,final_train_model_list,final_train_category_list,upsample)

        test_img_list = []
        test_model_list = []
        test_category_list = []

        for i in test:
            if not(y[i]['category']=='misc' or y[i]['category']=='tool'):
                test_img_list.append(y[i]['img'])
                test_model_list.append(y[i]['model' if self.config.is_mesh else 'voxel'])
                test_category_list.append(y[i]['category']+'_pix3d')

        return train_img_list,train_model_list,test_img_list,test_model_list,train_category_list,test_category_list

    def get_train_test_split_json(self,train_json_path,test_json_path, upsample=False):
        categories = json.load(open(train_json_path))['categories']
        category_list = [categories[i]['name'] for i in range(len(categories))]

        train_annotations = json.load(open(train_json_path))['annotations']
        test_annotations = json.load(open(test_json_path))['annotations']
        train_images = json.load(open(train_json_path))['images']
        test_images = json.load(open(test_json_path))['images']

        final_train_img_list = []
        final_train_model_list = []
        final_train_category_list = []

        for category in set(category_list):
            if category =='misc' or category =='tool':
                continue

            train_img_list = []
            train_model_list= []
            train_category_list = []

            for i in range(len(train_annotations)):
                if category_list[train_annotations[i]['category_id']-1]==category:
                    train_img_list.append(train_images[i]['img'])
                    train_model_list.append(train_annotations[i]['model' if self.config.is_mesh else 'voxel'])
                    train_category_list.append(category_list[train_annotations[i]['category_id']-1]+'_pix3d')

            final_train_img_list.append(train_img_list)
            final_train_model_list.append(train_model_list)
            final_train_category_list.append(train_category_list)

        train_img_list,train_model_list,train_category_list = self.upsampling(final_train_img_list,final_train_model_list,final_train_category_list,upsample)


        test_img_list = []
        test_model_list = []
        test_category_list = []
        for i in range(len(test_annotations)):
            if not(category_list[test_annotations[i]['category_id']-1]=='misc' or category_list[test_annotations[i]['category_id']-1]=='tool'):
                test_img_list.append(test_images[i]['img'])
                test_model_list.append(test_annotations[i]['model' if self.config.is_mesh else 'voxel'])
                test_category_list.append(category_list[test_annotations[i]['category_id']-1]+'_pix3d')


        return train_img_list,train_model_list,test_img_list,test_model_list,train_category_list,test_category_list

    def upsampling(self,final_train_img_list,final_train_model_list,final_train_category_list,upsample):
        # print([len(cat_list) for cat_list in final_train_category_list])
        num_to_upsample = max([len(cat_list) for cat_list in final_train_category_list])

        train_img_list = []
        train_model_list= []
        train_category_list = []
        for i in range(len(final_train_category_list)):
            img_list,model_list, category_list = resample(final_train_img_list[i],final_train_model_list[i], final_train_category_list[i], random_state=123,
                                                      n_samples= num_to_upsample if upsample else len(final_train_category_list[i]))
            train_img_list = train_img_list + img_list
            train_model_list = train_model_list + model_list
            train_category_list = train_category_list + category_list

        return train_img_list,train_model_list,train_category_list


    def get_train_test_indices(self,train_indices_path,test_indices_path):
        TRAIN_SPLIT_IDX = os.path.join(os.path.dirname(__file__),train_indices_path)
        TEST_SPLIT_IDX = os.path.join(os.path.dirname(__file__),test_indices_path)
        train = np.load(TRAIN_SPLIT_IDX)
        test = np.load(TEST_SPLIT_IDX)
        return train, test



class Pix3d(Dataset):
    def __init__(self,config,input_paths,output_paths, transforms=None, imagesPerCategory=0):
        self.input_paths = input_paths
        self.output_paths = output_paths

        self.config = config
        #paths
        self.dataset_path = config.dataset_path

        self.resize = config.resize
        self.size = config.size
        self.transform = transforms

        if imagesPerCategory != 0:
            self.init_nimages_per_category(imagesPerCategory)


    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path,self.input_paths[idx])
        taxonomy_id = self.input_paths[idx].split('/')[1] # img/bed/0008.png = bed
        if self.config.is_mesh:
            output_model_path = os.path.join(self.dataset_path,self.output_paths[idx])
        else:
            output_model_path = os.path.join(self.dataset_path,self.output_paths[idx])
            output_model_path = os.path.dirname(os.path.abspath(output_model_path))
            if self.config.label_type == Baseconfig.LabelType.NPY:
                output_model_path = os.path.join(output_model_path,"voxel_"+str(self.config.voxel_size)+".npy")
                # output_model_path.replace("voxel.mat","voxel_"+str(self.config.voxel_size)+".npy")
        # print(str(idx)+":img_path:"+img_path)
        # print(str(idx)+":output_model_path:"+output_model_path)

        input_img = self.read_img(img_path)
        input_stack = input_img

        # if self.config.include_depthmaps:
        #     depthmap_path = os.path.join(self.dataset_depthmap_path,self.input_paths[idx])
        #     input_depth = self.read_img(depthmap_path,type='L')
        #     input_stack = np.vstack((input_stack, input_depth))

        if self.config.include_masks:
            mask_path = os.path.join(self.dataset_path,self.input_paths[idx])
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
            img = cv2.resize(img, (self.size[0],self.size[1]), interpolation = cv2.INTER_AREA)/ 255.


        if len(img.shape) < 3:
            print('[WARN] It seems the image file %s is grayscale.' % (path))
            img = np.stack((img, ) * 3, -1)
        elif img.shape[2] > 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img = np.asarray([img])
        return img

    def __len__(self):
        # if self.config.platform == "darwin":
        #     return 16 #debug
        return len(self.input_paths)

    def unique_labels(self):
        taxonomies = []
        for i in range(0,self.__len__()):
            taxonomies.append(self.input_paths[i].split('/')[1])

        unique_taxonomy = list(set(taxonomies))
        return unique_taxonomy

    def init_nimages_per_category(self, num):
        inputPaths = []
        outputPaths = []
        for taxonomy in self.unique_labels():
            count = 0
            for i in range(0, self.__len__()):
                taxonomy_id = self.input_paths[i].split('/')[1] # img/bed/0008.png = bed
                if(taxonomy_id==taxonomy):
                    count+=1

                    inputPaths.append(self.input_paths[i])
                    outputPaths.append(self.output_paths[i])


                    if count == num:
                        break

        self.input_paths = inputPaths
        self.output_paths = outputPaths


class Pix3d_Img(Dataset):
    def __init__(self, config, input_paths, labels, customtransforms=None, imagesPerCategory=0):
        self.input_paths = input_paths
        self.labels = labels
        self.config = config

        #paths
        self.dataset_path = config.dataset_path

        self.resize = config.resize
        self.size = config.size
        self.transform = customtransforms

        if imagesPerCategory != 0:
            self.init_nimages_per_category(imagesPerCategory)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path,self.input_paths[idx])

        input_img = self.read_img(img_path)
        input_stack = self.transform(input_img)

        return input_stack, self.labels[idx]

    def read_img(self, path, type='RGB'):
        img =Image.open(path).convert("RGB")
        return img

    def init_nimages_per_category(self, num):
        inputPaths = []
        labels = []
        for taxonomy in self.unique_labels():
            count = 0
            for i in range(0, self.__len__()):
                taxonomy_id = self.labels[i]
                if(taxonomy_id==taxonomy):
                    count+=1

                    inputPaths.append(os.path.join(self.dataset_path,self.input_paths[i]))
                    labels.append(self.labels[i])

                    if count == num:
                        break

        self.input_paths = inputPaths
        self.labels = labels

    def unique_labels(self):
        taxonomies = []
        for i in range(0,self.__len__()):
            taxonomies.append(self.labels[i])
        unique_taxonomy = set(taxonomies)
        return unique_taxonomy

    def __len__(self):
        # return 10
        return len(self.input_paths)


if __name__ == '__main__':
    config = Baseconfig.config

    train_transforms = transforms.Compose([
        transform_utils.Normalize(mean=config.DATASET.MEAN, std=config.DATASET.STD),
        transform_utils.ToTensor(),
    ])
    config.dataset_path ='/Users/apple/OVGU/Thesis/Dataset/pix3d'
    config.pix3d.train_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_s1_train.json'
    config.pix3d.test_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_s1_test.json'
    config.upsample = True
    traindataset = Pix3dDataset(config).get_trainset(train_transforms)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=5, shuffle=True,num_workers=8)
    print(traindataset.__len__())
    print(traindataset.unique_labels())
    testdataset = Pix3dDataset(config).get_testset(train_transforms)
    print(testdataset.__len__())

    config.pix3d.train_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_s1_train.json'
    config.pix3d.test_indices = config.root_path + '/Datasets/pix3d/splits/pix3d_s1_test.json'
    config.upsample = False
    traindataset2 = Pix3dDataset(config).get_trainset(train_transforms)
    train_loader2 = torch.utils.data.DataLoader(traindataset, batch_size=5, shuffle=True,num_workers=8)
    print(traindataset2.__len__())
    print(traindataset2.unique_labels())
    testdataset2 = Pix3dDataset(config).get_testset(train_transforms)
    print(testdataset2.__len__())

    for batch_index, (taxonomy_id,input_batch, input_label) in enumerate(train_loader):
        # input_batch, input_label = input_batch[:, None, :].cuda(), input_label[:, None, :].cuda()
        # print(input_batch.shape)
        # print(input_label.shape)

        # print(input_batch[0][0:3].shape)
        print(taxonomy_id)
        break





