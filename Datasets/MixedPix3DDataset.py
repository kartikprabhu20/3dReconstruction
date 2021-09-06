"""

    Created on 05/09/21 11:32 PM 
    @author: Kartik Prabhu

"""
import torch

import Baseconfig
import transform_utils
from Datasets.pix3dsynthetic.BaseDataset import BaseDataset
import json
import os
import numpy as np
import cv2
from Datasets.pix3dsynthetic.SyntheticPix3dDataset import SyntheticPix3dDataset
from utils import mat_to_array, load
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.utils import resample


class MixedPix3DDataset(SyntheticPix3dDataset):
    def __init__(self,config, logger=None):
        self.config=config
        self.logger = logger

        if self.config.pix3d.train_indices.endswith('.npy'):
            self.pix3d_train_img_list,self.pix3d_train_model_list,self.pix3d_test_img_list, \
            self.pix3d_test_model_list,self.pix3d_train_category_list,self.pix3d_test_category_list \
                = self.get_train_test_split_npy(self.config.mixedtrain.real_dataset_path + "/pix3d.json", self.config.pix3d.train_indices,
                                                self.config.pix3d.test_indices, upsample=self.config.upsample)
        else:
            self.pix3d_train_img_list,self.pix3d_train_model_list,self.pix3d_test_img_list, \
            self.pix3d_test_model_list,self.pix3d_train_category_list,self.pix3d_test_category_list \
                = self.get_train_test_split_json(self.config.pix3d.train_indices,self.config.pix3d.test_indices, upsample=self.config.upsample)


        self.synth_train_img_list, self.synth_train_model_list,\
        self.synth_test_img_list, self.synth_test_model_list = self.get_train_test_split(
            self.config.s2r3dfree.train_test_split)

    def get_trainset(self, transforms=None, images_per_category=0):
        return MixedPix3D(self.config, self.pix3d_train_img_list, self.pix3d_train_model_list,
                          self.synth_train_img_list, self.synth_train_model_list,
                          transforms=transforms, imagesPerCategory=images_per_category, logger=self.logger)

    def get_testset(self, transforms=None, images_per_category=0):
        return MixedPix3D(self.config, self.pix3d_test_img_list, self.pix3d_test_model_list,
                          self.synth_test_img_list, self.synth_test_model_list,
                          transforms=transforms, imagesPerCategory=images_per_category, logger=self.logger)

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

    def get_train_test_indices(self,train_indices_path,test_indices_path):
        TRAIN_SPLIT_IDX = os.path.join(os.path.dirname(__file__),train_indices_path)
        TEST_SPLIT_IDX = os.path.join(os.path.dirname(__file__),test_indices_path)
        train = np.load(TRAIN_SPLIT_IDX)
        test = np.load(TEST_SPLIT_IDX)
        return train, test

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


class MixedPix3D(Dataset):
    def __init__(self, config, real_input_paths, real_output_paths, synth_input_paths, synth_output_paths, transforms=None, imagesPerCategory=0,logger=None):

        self.config = config
        self.logger=logger

        self.real_input_paths,self.real_output_paths  = self.shuffle(real_input_paths,real_output_paths)
        self.synth_input_paths,self.synth_output_paths  = self.shuffle(synth_input_paths,synth_output_paths)

        # print("lengths")
        # print(len(self.real_input_paths))
        # print(len(self.synth_input_paths))

        # print("paths 0:")
        # print(self.real_input_paths[0])
        # print(self.real_output_paths[0])
        # print(self.synth_input_paths[0])
        # print(self.synth_output_paths[0])

        # paths
        self.real_dataset_path = config.mixedtrain.real_dataset_path
        self.real_dataset_img_path = os.path.join(self.real_dataset_path, "imgs")
        # self.real_dataset_depthmap_path = os.path.join(self.real_dataset_path, "depthmaps")
        # self.real_dataset_masks_path = os.path.join(self.real_dataset_path, "masks")
        self.real_dataset_models_path = os.path.join(self.real_dataset_path, "models")

        self.synth_dataset_path = config.mixedtrain.synth_dataset_path
        self.synth_dataset_img_path = os.path.join(self.synth_dataset_path, "imgs")
        # self.synth_dataset_depthmap_path = os.path.join(self.synth_dataset_path, "depthmaps")
        # self.synth_dataset_masks_path = os.path.join(self.synth_dataset_path, "masks")
        self.synth_dataset_models_path = os.path.join(self.synth_dataset_path, "models")

        self.resize = config.resize
        self.size = config.size
        self.transform = transforms
        self.ratio =  self.config.mixedtrain.ratio


        # if imagesPerCategory != 0:
        #     self.init_nimages_per_category(imagesPerCategory)

    def shuffle(self, list1, list2):
        a = np.array(list1)
        b = np.array(list2)
        indices = np.arange(a.shape[0])
        np.random.shuffle(indices)

        return a[indices], b[indices]

    def __getitem__(self, idx):
        # print("idx:"+str(idx))
        if (idx) % self.config.batch_size < ((1-self.ratio) * self.config.batch_size):
            index  = idx - int(self.config.batch_size  * self.ratio) * int((idx)/self.config.batch_size)
            index = index % len(self.synth_input_paths)
            # print("synth index:" + str(index))
            img_path = os.path.join(self.synth_dataset_img_path, self.synth_input_paths[index])
            taxonomy_id = self.synth_input_paths[index].split('/')[0]
            output_model_path = os.path.join(os.path.join(self.synth_dataset_models_path, self.synth_output_paths[index]),
                                             "voxel_" + str(self.config.voxel_size) + ".npy" if self.config.label_type == Baseconfig.LabelType.NPY else "voxel.mat")

        else:
            index  = idx - int(self.config.batch_size  * (1-self.ratio)) * int(1+(idx)/self.config.batch_size)
            index = index % len(self.real_input_paths)
            # print("real index:" + str(index))

            img_path = os.path.join(self.real_dataset_path,self.real_input_paths[index])
            taxonomy_id = self.real_input_paths[index].split('/')[1] # img/bed/0008.png = bed
            output_model_path = os.path.join(self.real_dataset_path,self.real_output_paths[index])
            output_model_path = os.path.dirname(os.path.abspath(output_model_path))
            if self.config.label_type == Baseconfig.LabelType.NPY:
                output_model_path = os.path.join(output_model_path,"voxel_"+str(self.config.voxel_size)+".npy")

        # print(img_path)
        # print(taxonomy_id)
        # print(output_model_path)

        # if self.logger != None:
        #     self.logger.info(str(idx)+":img_path:"+img_path)
        #     self.logger.info(str(idx)+":output_model_path:"+output_model_path)

        input_img = self.read_img(img_path)
        input_stack = input_img

        # if self.config.include_depthmaps:
        #     depthmap_path = os.path.join(self.dataset_depthmap_path, self.input_paths[idx])
        #     input_depth = self.read_img(depthmap_path, type='L')
        #     input_stack = np.vstack((input_stack, input_depth))
        #
        # if self.config.include_masks:
        #     mask_path = os.path.join(self.dataset_masks_path, self.input_paths[idx])
        #     input_mask = self.read_img(mask_path, type='L')
        #     input_stack = np.vstack((input_stack, input_mask))

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

        return taxonomy_id, input_stack, output_model

    def unique_labels(self):
        taxonomies = []
        for i in range(0, self.__len__()):
            taxonomies.append(self.input_paths[i].split('/')[0])

        unique_taxonomy = list(set(taxonomies))
        return unique_taxonomy

    def init_nimages_per_category(self, num):
        inputPaths = []
        outputPaths = []
        for taxonomy in self.unique_labels():
            count = 0
            for i in range(0, self.__len__()):
                taxonomy_id = self.input_paths[i].split('/')[0]
                if (taxonomy_id == taxonomy):
                    count += 1

                    inputPaths.append(self.input_paths[i])
                    outputPaths.append(self.output_paths[i])

                    if count == num:
                        break

        self.input_paths = inputPaths
        self.output_paths = outputPaths

    def read_img(self, path, type='RGB'):
        # with Image.open(path) as img:
        #     img = img.convert(type)# convert('L') if it's a gray scale image
        #     if self.resize:
        #         img = img.resize((self.size[0],self.size[1]), Image.ANTIALIAS)

        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            if self.resize:
                img = cv2.resize(img, (self.size[0],self.size[1]), interpolation = cv2.INTER_AREA)/ 255.


            if len(img.shape) < 3:
                print('[WARN] It seems the image file %s is grayscale.' % (path))
                img = np.stack((img, ) * 3, -1)
            elif img.shape[2] > 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        except:
            self.logger.info("error in path: "+ path)

        img = np.asarray([img])
        return img

    def __len__(self):
        # if self.config.platform == "darwin":
        #     return 8 #debug
        length = len(self.real_input_paths)+len(self.synth_input_paths)
        return length

if __name__ == '__main__':
    config = Baseconfig.config

    IMG_SIZE = config.size[0],config.size[1]
    CROP_SIZE = config.crop_size[0], config.crop_size[1]
    train_transforms = transform_utils.Compose([
        transform_utils.RandomCrop(IMG_SIZE, CROP_SIZE),
        # transform_utils.RandomBackground(self.config.TRAIN.RANDOM_BG_COLOR_RANGE),
        transform_utils.ColorJitter(config.TRAIN.BRIGHTNESS, config.TRAIN.CONTRAST, config.TRAIN.SATURATION),
        transform_utils.RandomNoise(config.TRAIN.NOISE_STD),
        transform_utils.Normalize(mean=config.DATASET.MEAN, std=config.DATASET.STD),
        transform_utils.RandomFlip(),
        transform_utils.RandomPermuteRGB(),
        transform_utils.ToTensor(),
    ])

    train_transforms = transform_utils.Compose([
        transform_utils.CenterCrop(IMG_SIZE, CROP_SIZE),
        # transform_utils.RandomBackground(self.config.TEST.RANDOM_BG_COLOR_RANGE),
        transform_utils.Normalize(mean=config.DATASET.MEAN, std=config.DATASET.STD),
        transform_utils.ToTensor(),
    ])

    traindataset = MixedPix3DDataset(config).get_trainset(transforms=train_transforms)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=1)
    print(traindataset.__len__())
    print(train_loader.__len__())

    for batch_index, (taxonomy, input_batch, input_label) in enumerate(train_loader):
        print("====================batch:"+ str(batch_index))
        # input_batch, input_label = input_batch[:, None, :].cuda(), input_label[:, None, :].cuda()
        # print(input_batch.shape)
        # print(input_label.shape)
        #
        # print(input_batch[0][0:3].shape)
        # print(taxonomy)