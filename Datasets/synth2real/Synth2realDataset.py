"""

    Created on 04/09/21 12:00 AM 
    @author: Kartik Prabhu

"""
import os
import pickle
import random
from PIL import Image
import json

from Datasets.pix3dsynthetic.BaseDataset import BaseDataset


class Synth2RealDataset(BaseDataset):

    def __init__(self, config, logger=None):
        print("SyntheticPix3dDataset")
        self.logger = logger
        self.config = config
        # self.train_img_list,y, self.test_img_list,yt = self.get_train_test_split(self.config.SYNTH2REAL.syntheticsplit)
        # # self.train_out_img_list, self.test_out_img_list =
        #
        # self.train_out_img_list,y,self.test_out_img_list, \
        # yt,self.train_category_list,self.test_category_list \
        #     = self.get_train_test_split_json(self.config.pix3d.train_indices,self.config.pix3d.test_indices, upsample=self.config.upsample)

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

    def get_train_test_split(self, filePath):
        pickle_file = pickle.load(open(filePath, "rb"))
        x = pickle_file['train_x']
        y = pickle_file['train_y']
        xt = pickle_file['test_x']
        yt = pickle_file['test_y']

        return x, y, xt, yt

    def get_trainset(self, transforms=None, images_per_category=0):
        return Synth2Real(input_paths = self.config.SYNTH2REAL.inputPath, output_paths = self.config.SYNTH2REAL.outputpath , transforms=transforms)

    # def get_testset(self, transforms=None, images_per_category=0):
    #     return Synth2Real(input_paths = self.test_img_list,output_paths= self.test_out_img_list, transforms=transforms,)

class Synth2Real():

    # def __init__(self,input_paths, output_paths, transforms=None):
    #     self.categoryInputArrays = []
    #     self.categoryOutputArrays = []
    #
    #     print(input_paths[0])
    #     print(output_paths[0])


    def __init__(self,input_paths, output_paths, transforms=None):
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.transform = transforms

        print(self.get_classes(input_paths))

        self.categoryInputArrays = []
        self.categoryOutputArrays = []

        for label in self.get_classes(input_paths):

            labelImages = []

            labelPath = os.path.join(input_paths,label)
            for folder in os.listdir(labelPath):
                folderpath = os.path.join(labelPath,folder)
                if(folder == ".DS_Store"):
                    continue

                for img in os.listdir(folderpath):
                    imgPath = os.path.join(folderpath,img)
                    labelImages.append(imgPath)

            self.categoryInputArrays.append(labelImages)

        for label in self.get_classes(output_paths):

            labelImages = []

            labelPath = os.path.join(output_paths,label)
            for img in os.listdir(labelPath):
                imgPath = os.path.join(labelPath,img)
                labelImages.append(imgPath)

            self.categoryOutputArrays.append(labelImages)


    def __getitem__(self, idx):

        index = idx%7

        input_img = self.read_img(self.categoryInputArrays[index][random.randint(0, len(self.categoryInputArrays[index]) - 1)])
        input_stack = self.transform(input_img)

        output_img = self.read_img(self.categoryOutputArrays[index][random.randint(0, len(self.categoryOutputArrays[index]) - 1)])
        output_stack = self.transform(output_img)

        return input_stack, output_stack


    def read_img(self, path, type='RGB'):
        img = Image.open(path)
        img = img.convert('RGB')
        return img

    def __len__(self):
        length = 0
        for labels in self.categoryInputArrays:
            length += len(labels)
        return length

    def get_classes(self,root_path):
        return self.get_classes_path(root_path)

    def get_classes_path(self,path):
        return [class_folder for class_folder in os.listdir(path) if not class_folder.startswith('.')]
