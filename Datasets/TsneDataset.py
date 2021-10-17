"""

    Created on 27/09/21 9:28 AM 
    @author: Kartik Prabhu

"""
import json
import pickle

from PIL import Image
from torch.utils.data import Dataset
import os

from Datasets.pix3d.Pix3dDataset import Pix3dDataset
from Datasets.pix3dsynthetic.SyntheticPix3dDataset import SyntheticPix3dDataset


class ChairPix3dDataset(Pix3dDataset):
    def get_img_trainset(self, transforms=None,images_per_category=0):
        return ChairPix3d(self.config,self.train_img_list,self.train_category_list,customtransforms=transforms,imagesPerCategory=images_per_category)

    def get_img_testset(self,transforms=None,images_per_category=0):
        return ChairPix3d(self.config,self.test_img_list,self.test_category_list,customtransforms=transforms,imagesPerCategory=images_per_category)

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
            if (y[i]['category']=='chair' ):
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

class ChairPix3d(Dataset):

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


class SyntheticChairDataset(SyntheticPix3dDataset):
    def __init__(self, config, logger=None):
        super().__init__(config, logger)

        self.train_chair_list = []
        for path in self.train_img_list:
            if 'chair'== path.split('/')[0]:
                self.train_chair_list.append(path)

        self.test_chair_list = []
        for path in self.test_img_list:
            if 'chair'== path.split('/')[0]:
                self.test_chair_list.append(path)


    def get_img_trainset(self, transforms=None, images_per_category=0):
        return SyntheticChair_img(self.config, self.train_chair_list, customtransforms=transforms,
                                  imagesPerCategory=images_per_category)

    def get_img_testset(self, transforms=None, images_per_category=0):
        return SyntheticChair_img(self.config, self.test_chair_list, customtransforms=transforms,
                                  imagesPerCategory=images_per_category)

    def get_train_test_split(self, filePath):
        pickle_file = pickle.load(open(filePath, "rb"))
        x = pickle_file['train_x']
        y = pickle_file['train_y']
        xt = pickle_file['test_x']
        yt = pickle_file['test_y']

        return x, y, xt, yt

class SyntheticChair_img(Dataset):
    def __init__(self, config, input_paths, customtransforms=None, imagesPerCategory=0):
        self.dataset_path = config.dataset_path
        self.dataset_img_path = os.path.join(self.dataset_path, "imgs")

        self.input_paths = input_paths
        self.config = config

        # paths
        self.dataset_path = config.dataset_path

        self.resize = config.resize
        self.size = config.size
        self.transform = customtransforms

        if imagesPerCategory != 0:
            self.init_nimages_per_category(imagesPerCategory)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_img_path, self.input_paths[idx])
        taxonomy_id = self.input_paths[idx].split('/')[0]
        input_img = self.read_img(img_path)
        input_stack = self.transform(input_img) if not self.transform == None else input_img

        return input_stack, taxonomy_id

    def read_img(self, path, type='RGB'):
        img = Image.open(path).convert("RGB")
        return img

    def init_nimages_per_category(self, num):
        inputPaths = []
        labels = []
        for taxonomy in self.unique_labels():
            count = 0
            for i in range(0, self.__len__()):
                taxonomy_id = self.input_paths[i].split('/')[0]  # img/bed/0008.png = bed
                if (taxonomy_id == taxonomy):
                    count += 1

                    inputPaths.append(self.input_paths[i])

                    if count == num:
                        break

        self.input_paths = inputPaths

    def unique_labels(self):
        taxonomies = []
        for i in range(0, self.__len__()):
            taxonomies.append(self.input_paths[i].split('/')[0])
        unique_taxonomy = set(taxonomies)
        return unique_taxonomy

    def __len__(self):
        # return 10
        return len(self.input_paths)