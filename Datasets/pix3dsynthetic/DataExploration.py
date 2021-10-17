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
    return get_classes_path(root_path+"/models/")

def get_classes_path(path):
    #return ["bed","bookcase","chair","desk","sofa","table","wardrobe"]
    return [class_folder for class_folder in os.listdir(path) if not class_folder.startswith('.')]

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

    for label in get_classes_path(os.path.join(root_path,"imgs")):
        class_array = []
        model_array = []

        labelPath = os.path.join(root_path+"models/", label)
        for folder in os.listdir(labelPath):
            if folder =='.DS_Store' or folder =='._.DS_Store':
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


def save_train_test_split(xtrain, ytrain, xtest, ytest, fileNmae = "train_test_split.p"):
    file_name = fileNmae

    open_file = open(file_name, "wb")
    pickle.dump({'train_x':xtrain, 'train_y': ytrain, 'test_x': xtest, 'test_y': ytest}
                ,open(file_name, "wb"))
    open_file.close()

def save_model_split(models_train, labels_train, models_test, labels_test, file_name):
    '''
    example of models:'IKEA_DOMBAS', 'IKEA_PAX_2', 'IKEA_HEMNES'
    example of labels:'wardrobe', 'chair', 'desk'

    :param models_train:
    :param labels_train:
    :param models_test:
    :param labels_test:
    :return:
    '''
    file_name = file_name
    open_file = open(file_name, "wb")
    pickle.dump({'models_train':models_train, 'labels_train': labels_train,
                 'models_test':models_test, 'labels_test': labels_test}
            ,open(file_name, "wb"))
    open_file.close()

def generate_train_test_split(root_path,model_train,label_train,models_test,labels_test):

    train_imgs = []
    train_output = []

    test_imgs = []
    test_output = []

    imgs_path = os.path.join(root_path,"imgs")
    for i in range(0,len(model_train)):
        model_folder = label_train[i]+"/"+model_train[i]
        for file in os.listdir(os.path.join(imgs_path,model_folder)):
            train_imgs.append(model_folder+'/'+file)
            train_output.append(model_folder)

    for i in range(0,len(models_test)):
        model_folder = labels_test[i]+"/"+models_test[i]
        for file in os.listdir(os.path.join(imgs_path,model_folder)):
            test_imgs.append(model_folder+'/'+file)
            test_output.append(model_folder)

    return train_imgs, train_output, test_imgs, test_output

def generate_train_test_split_modelwise(root_path,model_train,label_train,models_test,labels_test):

    train_imgs = []
    train_output = []

    test_imgs = []
    test_output = []

    imgs_path = os.path.join(root_path,"imgs")
    for i in range(0,len(model_train)):
        img = label_train[i]+"/"+model_train[i]
        train_imgs.append(img)
        train_output.append( label_train[i]+"/"+model_train[i].split('/')[0])

    for i in range(0,len(models_test)):
        img = labels_test[i]+"/"+models_test[i]
        test_imgs.append(img)
        test_output.append(labels_test[i]+"/"+models_test[i].split('/')[0])

    return train_imgs, train_output, test_imgs, test_output

def get_model_split(filePath):
    pickle_file = pickle.load(open(filePath, "rb" ))
    x = pickle_file['models_train']
    y = pickle_file['labels_train']
    xt = pickle_file['models_test']
    yt = pickle_file['labels_test']

    return  x,y,xt,yt

def get_train_test_split(filePath):
    pickle_file = pickle.load(open(filePath, "rb" ))
    x = pickle_file['train_x']
    y = pickle_file['train_y']
    xt = pickle_file['test_x']
    yt = pickle_file['test_y']

    return x,y,xt,yt

def get_train_test_histogram(root_path,train_img_list,test_img_list,fileName = "test"):
    labels = get_classes(root_path)

    train_list = [0] * len(get_classes(root_path))
    test_list = [0] * len(get_classes(root_path))

    for i in range(len(train_img_list)):
        train_list[labels.index(train_img_list[i].split('/')[0])] += 1

    for i in range(len(test_img_list)):
        test_list[labels.index(test_img_list[i].split('/')[0])] += 1

    xticks = [i for i in range(len(labels))]
    plt.bar(xticks, train_list, color='b', width=0.9)
    plt.bar(xticks, test_list, bottom=train_list, color='g', width=0.9)
    plt.xticks(xticks, labels)
    plt.title("Train-Test split of Models in S2R:3DFREE_Chair_final")
    plt.legend(['train','test'])
    plt.savefig(fileName,format='pgf')
    # plt.show()


def get_all_model_names(root_path):

    model_path = os.path.join(root_path,"models")
    model_names = set()
    for label in get_classes(root_path):
        labelPath = os.path.join(model_path,label)
        for folder in os.listdir(labelPath):
            model_names.add(label+"/"+folder)

    # print(model_names)
    # print(len(model_names))
    return model_names

def get_model_names_histogram(root_path):
    imgs_path = os.path.join(root_path,"imgs")
    model_names = list(get_all_model_names(root_path))
    model_list = [0] * len(model_names)

    for label in get_classes_path(imgs_path):
        labelPath = os.path.join(imgs_path,label)
        for folder in os.listdir(labelPath):
            model_list[model_names.index(folder)] += len(os.listdir(os.path.join(labelPath,folder)))

    xticks = [i for i in range(len(model_names))]
    plt.bar(xticks, model_list, color='b')
    plt.title("Models in s2r3dfree")
    # plt.xticks(xticks, model_names)
    # plt.legend(['Mode'])
    plt.show()

def get_all_models_grouped(root_path):
    models_list = []
    labels_list = []

    model_path = os.path.join(root_path,"imgs")
    for label in get_classes_path(os.path.join(root_path,"imgs")) :
        labelPath = os.path.join(model_path,label)
        for model in os.listdir(labelPath):
            if model == ".DS_Store":
                continue
            model_indices = [model+'/'+file for file in os.listdir(os.path.join(labelPath,model))]
            labels = [label for file in os.listdir(os.path.join(labelPath,model))]
            models_list.append(model_indices)
            labels_list.append(labels)
    # print(indices_list)
    return labels_list,models_list

def get_train_test_model_names_histogram(root_path,train_model_list,test_model_list,fileName="test"):

    model_names = list(get_all_model_names(root_path))

    print(len(list(set(train_model_list) & set(test_model_list)))) #confirm no intersections

    train_list = [0] * len(model_names)
    test_list = [0] * len(model_names)

    for model in train_model_list:
        train_list[model_names.index(model)] += 1

    for model in test_model_list:
        test_list[model_names.index(model)] += 1

    print(train_list)
    print(test_list)

    xticks = [i for i in range(len(model_names))]
    plt.bar(xticks, train_list, color='b')
    plt.bar(xticks, test_list, bottom=train_list, color='g')
    # plt.xticks(xticks, model_names)
    plt.title("Train-Test split of Chair Models in S2R:3DFREE")
    plt.legend(['train','test'])
    plt.savefig(fileName,format='pgf')
    # plt.show()

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
    # root_path = '/Users/apple/OVGU/Thesis/SynthDataset1/'
    # label_list, model_list = get_all_models_classwise(root_path)
    # x,y,xt,yt = split_dataset(model_list,label_list)
    # print(x)
    # print(y)
    # print(xt)
    # print(yt)
    # # print(len(x))
    # # print(len(xt))
    # save_model_split(x,y,xt,yt)
    #
    # model_train,label_train,models_test,labels_test = get_model_split("model_split.p") # gives only model, to get images of each model next line
    # print(model_train)
    # print(label_train)
    # print(models_test)
    # print(labels_test)
    # x,y,xt,yt = generate_train_test_split(root_path,model_train,label_train,models_test,labels_test)
    # print(x)
    # print(y)
    # # print(xt)
    # print(len(x))
    # print(len(xt))
    # save_train_test_split(x,y,xt,yt)
#########################################################


    # train_img_list,train_model_list,test_img_list,test_model_list = get_train_test_split(
    #     "/Users/apple/OVGU/Thesis/code/3dReconstruction/Datasets/pix3dsynthetic/train_test_split.p")
    #
    # get_train_test_histogram(root_path,train_img_list,test_img_list)

#########################################################

    # root_path = '/Users/apple/OVGU/Thesis/s2r3dfree_v1/'
    # print(get_all_model_names(root_path))
    # get_model_names_histogram(root_path)


#########################################################
    # root_path = '/Volumes/Prabhu/thesis/s2r3dfree_v1/'
    # root_path = '/Users/apple/OVGU/Thesis/s2r3dfree_v3_2/'
    root_path = '/Users/apple/OVGU/Thesis/s2r3dfree_chair_light/'
    label_list, model_list = get_all_models_classwise(root_path)
    x,y,xt,yt = split_dataset(model_list,label_list)
    file_name = "splits/train_test_split_classwise_chair_light_10000.p"
    save_model_split(x,y,xt,yt, file_name)

    model_train,label_train,models_test,labels_test = get_model_split(file_name) # gives only model, to get images of each model next line
    x,y,xt,yt = generate_train_test_split(root_path,model_train,label_train,models_test,labels_test)
    save_train_test_split(x,y,xt,yt,"splits/train_test_split_classwise_chair_light_10000.p")

    train_img_list,train_model_list,test_img_list,test_model_list = get_train_test_split(
        "splits/train_test_split_classwise_chair_light_10000.p")
    print(train_img_list[0])
    print(train_model_list[0])
    # get_train_test_histogram(root_path,train_img_list,test_img_list,fileName='histogram_classwise_chair_final.pgf')
    # get_train_test_model_names_histogram(root_path,train_model_list,test_model_list,fileName='classwisewise_chair_final.pgf')

#########################################################

    label_list, img_list = get_all_models_grouped(root_path)
    x,y,xt,yt = split_dataset(img_list,label_list)
    x,y,xt,yt = generate_train_test_split_modelwise(root_path,x,y,xt,yt)
    save_train_test_split(x,y,xt,yt,"splits/train_test_split_modelwise_chair_light_10000.p")

    train_img_list,train_model_list,test_img_list,test_model_list = get_train_test_split(
        "splits/train_test_split_modelwise_chair_light_10000.p")

    print(len(train_model_list))
    print(len(test_model_list))


    # get_train_test_histogram(root_path,train_img_list,test_img_list,fileName='histogram_modelwise_chair_final.pgf')
    # get_train_test_model_names_histogram(root_path,train_model_list,test_model_list,fileName='modelwise_chair_final.pgf')

    #########################################################
