
"""

    Created on 05/01/21 6:34 PM
    @author: Kartik Prabhu

"""

import matplotlib.pyplot as plt
import json
import os
import numpy as np
import cv2

from utils import load_mat_pix3d


def getHistogram(root_path):
    labels = get_classes(root_path)
    xticks = [i for i in range(len(labels))]
    y = []
    for label in labels:
        y.append(len(os.listdir(os.path.join(root_path, label))))

    plt.bar(xticks, height=y)
    plt.xticks(xticks, labels)
    plt.show()

def get_occluded_Histogram(root_path):
    labels = get_classes(root_path)
    xticks = [i for i in range(len(labels))]
    y = []
    for label in labels:
        y.append(len(json.load(open(root_path + label+"_occluded.json"))))

    plt.bar(xticks, height=y)
    plt.xticks(xticks, labels)
    plt.show()

def get_classes(root_path):
    #return ["bed","bookcase","chair","desk","sofa","table","wardrobe"]
    return [class_folder for class_folder in os.listdir(root_path) if not class_folder.startswith('.')]

def split_to_classes(json_path, root_path):
    labels = get_classes(root_path)
    labels.append('misc') #not used
    labels.append('tool') #not used

    y = json.load(open(json_path))
    category = y[0]['category']

    j = 0 #pix3d.json is arranged in order or labels
    for label in labels: #iterate label number of times
        image_config = []
        for i in range(j, len(y)):
            if not category == y[i]['category']:
                j = i
                break
            image_config.append(y[i])

        with open(os.path.dirname(pix3d_json_path) + '/' + category + '.json', 'w') as fp:
            json.dump(image_config, fp)
        category = y[j]['category']


def save_occluded_json(root_path):
    labels = get_classes(root_path)

    for label in labels:
        y = json.load(open(root_path + label+".json"))
        occluded_config = []
        for data in y:
            if data["occluded"]:
                occluded_config.append(data)

        with open(root_path + label+"_occluded.json", 'w') as fp:
            json.dump(occluded_config, fp)

def get_train_test_split():
    TRAIN_SPLIT_IDX = os.path.join(os.path.dirname(__file__),
                                   'splits/pix3d_train.npy')
    TEST_SPLIT_IDX = os.path.join(os.path.dirname(__file__),
                                  'splits/pix3d_test.npy')
    train = np.load(TRAIN_SPLIT_IDX)
    test = np.load(TEST_SPLIT_IDX)
    return train, test

def get_train_test_histogram(root_path,json_path):
    y = json.load(open(json_path))
    train, test = get_train_test_split()
    labels = get_classes(root_path)

    train_list = [0] * len(get_classes(root_path))
    test_list = [0] * len(get_classes(root_path))

    for i in train:
        if not(y[i]['category']=='misc' or y[i]['category']=='tool'):
            train_list[labels.index(y[i]['category'])] += 1

    for i in test:
        if not(y[i]['category']=='misc' or y[i]['category']=='tool'):
            test_list[labels.index(y[i]['category'])] += 1

    xticks = [i for i in range(len(labels))]
    plt.bar(xticks, train_list, color='b', width=0.9)
    plt.bar(xticks, test_list, bottom=train_list, color='g', width=0.9)
    plt.xticks(xticks, labels)
    plt.legend(['train','test'])
    plt.show()

def save_edge_images(root_path, labels):
    for label in labels:
        edge_folder = os.path.join(root_path +"edge/", label)
        if not os.path.exists(edge_folder):
            os.makedirs(edge_folder)

    for label in labels:
        for filename in os.listdir(os.path.join(root_path +"img/", label)):
            img_path = os.path.join(root_path +"img/", label+"/") + filename
            edge_folder = os.path.join(root_path +"edge/", label)
            img = cv2.imread(img_path,0)
            edge_image = cv2.Canny(img,100,200)
            cv2.imwrite(os.path.join(edge_folder,filename), edge_image)

def get_voxel(root_path, category, image):
    y = json.load(open(os.path.join(root_path,category+".json")))
    for item in y:
        if os.path.basename(item["img"])==image:
            voxel_path = os.path.dirname(item["model"])
            return os.path.join(root_path, os.path.join(voxel_path,'voxel.mat'))
    return None

def get_model(root_path, category, image):
    y = json.load(open(os.path.join(root_path,category+".json")))
    for item in y:
        if os.path.basename(item["img"])==image:
            return os.path.join(root_path, item["model"])
    return None

if __name__ == '__main__':

    pix3d_json_path = '/Users/apple/OVGU/Thesis/Dataset/pix3d/pix3d.json'
    root_path = "/Users/apple/OVGU/Thesis/Dataset/pix3d/"

    # y = json.load(open(pix3d_json_path))

    #split the json into class
    # split_to_classes(pix3d_json_path,root_path)

    ##Histogram test
    # getHistogram("/Users/apple/OVGU/Thesis/Dataset/pix3d/img/")

    #Save occluded json
    # save_occluded_json(root_path)

    #Occluded Histogram
    # get_occluded_Histogram(root_path)

    #edge images
    # save_edge_images(root_path)

    # get_train_test_histogram(root_path,pix3d_json_path)

    #voxel_path
    # voxel_path = get_voxel(root_path,"chair","0002.png")
    voxel_path = '/Users/apple/OVGU/Thesis/SynthDataset1/models/chair/IKEA_HERMAN/voxel.mat'

    mesh_mat = load_mat_pix3d(voxel_path)
    mesh_mat.show()

    #mode_path
    # model_path = get_model(root_path,"chair","0002.png")
    # mesh_mat = load(model_path)
    # mesh_mat.show()



