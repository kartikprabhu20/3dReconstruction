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
from sklearn.model_selection import train_test_split
import pickle

def colored_depthmap(depth, d_min=None, d_max=None):
    # cmap = plt.cm.viridis
    cmap = plt.get_cmap('jet')
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def delete_unusedfile(root_path):
    '''
    Some files "CameraPositions_depth.png" ,"CameraPositions_img.png" ,"CameraPositions_layer.png" were blank and not needed
    '''
    for label in get_classes(root_path):
        labelPath = os.path.join(root_path +"model/", label)
        for folder in os.listdir(labelPath):
            if folder =='.DS_Store':
                continue
            imgFolderPath  = os.path.join(os.path.join(root_path +"model/", label), folder+'/img/')
            for filename in os.listdir(imgFolderPath):
                imgPath  = os.path.join(imgFolderPath,filename)
                if filename=="CameraPositions_depth.png" or filename == "CameraPositions_img.png" or filename =="CameraPositions_layer.png":
                    print(imgPath)
                    os.remove(imgPath)



def get_classes(root_path):
    #return ["bed","bookcase","chair","desk","sofa","table","wardrobe"]
    return [class_folder for class_folder in os.listdir(root_path+"/model/") if not class_folder.startswith('.')]

def getHistogram_fromarray(root_path, arrayList):
    labels = get_classes(root_path)
    xticks = [i for i in range(len(labels))]
    y = []
    for label in arrayList:
        y.append(len(label))

    plt.bar(xticks, height=y)
    plt.xticks(xticks, labels)
    plt.show()



def get_all_models_classwise(root_path):
    class_list = []
    models_list = []

    for label in get_classes(root_path):
        class_array = []
        model_array = []

        labelPath = os.path.join(root_path+"model/", label)
        for folder in os.listdir(labelPath):
            if folder =='.DS_Store':
                continue

            class_array.append(label)
            model_array.append(folder)

        class_list.append(class_array)
        models_list.append(model_array)

    return class_list, models_list



def get_all_filePaths_classwise(root_path):
    label_arrays = []
    model_arrays = []

    for label in get_classes(root_path):
        label_array = []
        model_array = []

        labelPath = os.path.join(root_path +"model/", label)
        for folder in os.listdir(labelPath):
            if folder =='.DS_Store':
                continue
            imgFolderPath  = os.path.join(os.path.join(root_path +"model/", label), folder+'/img/')
            modelPath = os.path.join(os.path.join(root_path +"model/", label),folder+"/model.obj")
            for filename in os.listdir(imgFolderPath):
                imgPath  = os.path.join(imgFolderPath,filename)
                if filename.__contains__("img"):
                    label_array.append(imgPath)
                    model_array.append(modelPath)

        label_arrays.append(label_array)
        model_arrays.append(model_array)

    return label_arrays, model_arrays

def split_dataset(input,output):

    final_train_img = []
    final_train_y = []

    final_test_img = []
    final_test_y = []

    for i in range(0,len(input)):
        x_train, x_test, y_train, y_test = train_test_split(input[i], output[i],train_size = 0.7, shuffle=True)
        final_train_img += x_train
        final_test_img += x_test
        final_train_y += y_train
        final_test_y += y_test

    return final_train_img,final_train_y,final_test_img,final_test_y


def save_train_test_split(xtrain, ytrain, xtest, ytest):
    file_name = "train_test_split.p"

    open_file = open(file_name, "wb")
    pickle.dump({'train_x':xtrain, 'train_y': ytrain, 'test_x': xtest, 'test_y': ytest}
                ,open(file_name, "wb"))
    open_file.close()

def save_model_split(models_train, labels_train, models_test, labels_test):
    file_name = "model_split.p"
    open_file = open(file_name, "wb")
    pickle.dump({'models_train':models_train, 'labels_train': labels_train,
                 'models_test':models_test, 'labels_test': labels_test}
            ,open(file_name, "wb"))
    open_file.close()

def get_train_test_split(root_path,model_train,label_train,models_test,labels_test):

    train_imgs = []
    train_output = []

    test_imgs = []
    test_output = []

    i = 0

    imgs_path = os.path.join(root_path,"imgs")
    for i in range(0,len(model_train)):
        model_folder = label_train[i]+"/"+model_train[i]
        for file in os.listdir(os.path.join(imgs_path,model_folder)):
            train_imgs.append(model_folder+'/'+file)
            train_output.append(model_folder)

    for i in range(0,len(models_test)):
        model_folder = label_train[i]+"/"+model_train[i]
        for file in os.listdir(os.path.join(imgs_path,model_folder)):
            test_imgs.append(model_folder+'/'+file)
            test_output.append(model_folder)

    return train_imgs, train_output, test_imgs, test_output

def get_model_split(filePath):
    pickle_file = pickle.load(open(filePath, "rb" ))
    x = pickle_file['models_train']
    y = pickle_file['labels_train']
    xt = pickle_file['models_test']
    yt = pickle_file['labels_train']

    return  x,y,xt,yt

def get_train_test_split(filePath):
    pickle_file = pickle.load(open(filePath, "rb" ))
    x = pickle_file['train_x']
    y = pickle_file['train_y']
    xt = pickle_file['test_x']
    yt = pickle_file['test_y']

    return x,y,xt,yt

if __name__ == '__main__':
    # depth = np.clip(np.load("/Users/apple/OVGU/Thesis/images/pix3dsynthetic/test_depth.png"), 1.0, 10.0) / 10 * 255
    # depth = Image.open("/Users/apple/OVGU/Thesis/images/synthetic/test_depth.png").convert('L')
    # # depth = Image.fromarray(depth.astype(np.uint8)[:,:], mode='L')
    # depth.show()
    #
    # depth = np.asarray(depth)
    # print(depth.shape)
    #
    # rgba_img = colored_depthmap(depth)
    # print(rgba_img.shape)
    # img_depth = Image.fromarray(rgba_img.astype(np.uint8))
    # img_depth.show()


#########################################################
    # root_path = '/Users/apple/OVGU/Thesis/Dataset/pix3d/'
    # # delete_unusedfile(root_path)
    #
    # print(get_classes(root_path))
    # input_arrays, model_arrays = get_all_filePaths_classwise(root_path)
    # # getHistogram_fromarray(root_path,input_arrays)
    #
    # x,y,xt,yt = split_dataset(input_arrays,model_arrays)
    # save_into_pickle( x,y,xt,yt)
    # print(len(x))
    #
    # x,y,xt,yt = get_train_test_split("train_test_split.p")
    # print(len(x))
    #
    # print(x[0])


#########################################################
    root_path = '/Users/apple/OVGU/Thesis/SynthDataset1/'
    # label_list, model_list = get_all_models_classwise(root_path)
    #
    # x,y,xt,yt = split_dataset(model_list,label_list)
    # save_model_split(x,y,xt,yt)

    # model_train,label_train,models_test,labels_test = get_model_split("model_split.p") # gives only model, to get images of each model next line
    # x,y,xt,yt = get_train_test_split(root_path,model_train,label_train,models_test,labels_test)
    # save_train_test_split(x,y,xt,yt)
#########################################################



