
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
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

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
    # return ["bed","bookcase","chair","desk","sofa","table","wardrobe"]
    return [class_folder for class_folder in os.listdir(root_path+'/model/') if not class_folder.startswith('.')]

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

def get_all_model_names(json_path):
    y = json.load(open(json_path))
    category = y[0]['category']

    model_names = set()
    for i in range(0, len(y)):
        if not(y[i]['category']=='misc' or y[i]['category']=='tool'):
            mode_name = y[i]['model'].split("/")[2]
            model_names.add(y[i]['category']+"/"+mode_name)

    print(model_names)
    print(len(model_names))
    return model_names

def get_model_names_histogram(json_path):
    y = json.load(open(json_path))
    model_names = list(get_all_model_names(json_path))
    model_list = [0] * len(model_names)

    for i in range(len(y)):
        if not(y[i]['category']=='misc' or y[i]['category']=='tool'):
             model_list[model_names.index(y[i]['category']+"/"+ y[i]['model'].split("/")[2])] += 1

    xticks = [i for i in range(len(model_names))]
    plt.bar(xticks, model_list, color='b')
    plt.title("Models in pix3D")
    # plt.xticks(xticks, model_names)
    # plt.legend(['Mode'])
    plt.show()

def get_all_indices_grouped(json_path):
    y = json.load(open(json_path))
    model_names = list(get_all_model_names(json_path))
    indices_list = []

    for model_name in model_names:
        model_indices = []
        for i in range(len(y)):
             if model_name.split("/")[1] == y[i]['model'].split("/")[2]:
                model_indices.append(i)

        indices_list.append(model_indices)
    # print(indices_list)
    return indices_list

def train_test_split_json(json_path):
    indices_list = get_all_indices_grouped(json_path)

    train_indices = []
    test_indices = []

    for i in range(0,len(indices_list)):
        # print(indices_list[i])
        # print(len(indices_list[i]))

        if len(indices_list[i]) < 2:
            train_indices += indices_list[i]
        else:
            x_train, x_test = train_test_split(indices_list[i],train_size = 0.7, shuffle=True)
            train_indices += x_train
            test_indices += x_test

    print(train_indices)
    print(test_indices)


    with open('/Users/apple/OVGU/Thesis/code/3dReconstruction/Datasets/pix3d/splits/pix3d_train_2.npy', 'wb') as f:
        np.save(f, np.array(train_indices))
    with open('/Users/apple/OVGU/Thesis/code/3dReconstruction/Datasets/pix3d/splits/pix3d_test_2.npy', 'wb') as f:
        np.save(f, np.array(test_indices))




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

def get_train_test_indices(train_file, test_file):
    train = np.load(train_file)
    test = np.load(test_file)
    return train, test

def get_train_test_model_names_histogram(json_path,train_file, test_file):
    y = json.load(open(json_path))
    model_names = list(get_all_model_names(json_path))
    train, test = get_train_test_indices(train_file, test_file)

    # print(len(train))
    # print(len(test))
    # print(len(model_names))

    #print(len(list(set(train) & set(test)))) #confirm no intersections

    train_list = [0] * len(model_names)
    test_list = [0] * len(model_names)

    for i in train:
        train_list[model_names.index(y[i]['category']+"/"+y[i]['model'].split("/")[2])] += 1

    for i in test:
        test_list[model_names.index(y[i]['category']+"/"+y[i]['model'].split("/")[2])] += 1

    print(len(train_list))
    print(len(test_list))

    xticks = [i for i in range(len(model_names))]
    plt.bar(xticks, train_list, color='b')
    plt.bar(xticks, test_list, bottom=train_list, color='g')
    plt.title("Train-Test split of Models in pix3D")
    # plt.xticks(xticks, model_names)
    plt.legend(['train','test'])
    plt.show()

def get_train_test_model_names_histogram_json(json_path,train_file, test_file):
    y = json.load(open(json_path))
    model_names = list(get_all_model_names(json_path))
    # train, test = get_train_test_indices(train_file, test_file)
    categories = json.load(open(train_file))['categories']
    category_list = [categories[i]['name'] for i in range(len(categories))]

    train_annotations = json.load(open(train_file))['annotations']
    test_annotations = json.load(open(test_file))['annotations']

    # print(len(train_annotations))
    # print(len(test_annotations))
    # print(len(model_names))
    # print(model_names)

    train_list = [0] * len(model_names)
    test_list = [0] * len(model_names)

    for i in range(len(train_annotations)):
        if not(category_list[train_annotations[i]['category_id']-1]=='misc' or category_list[train_annotations[i]['category_id']-1]=='tool'):
            train_list[model_names.index(category_list[train_annotations[i]['category_id']-1]+"/"+train_annotations[i]['model'].split("/")[2])] += 1

    for i in range(len(test_annotations)):
        if not(category_list[test_annotations[i]['category_id']-1]=='misc' or category_list[test_annotations[i]['category_id']-1]=='tool'):
            test_list[model_names.index(category_list[test_annotations[i]['category_id']-1] +"/"+ test_annotations[i]['model'].split("/")[2])] += 1

    xticks = [i for i in range(len(model_names))]
    plt.bar(xticks, train_list, color='b')
    plt.bar(xticks, test_list, bottom=train_list, color='g')
    plt.title("Train-Test split of Models in pix3D")
    # plt.xticks(xticks, model_names)
    plt.legend(['train','test'])
    plt.show()

def upsampling(final_train_img_list,final_train_model_list,final_train_category_list,upsample):
    # print([len(cat_list) for cat_list in final_train_category_list])
    num_to_upsample = max([len(cat_list) for cat_list in final_train_category_list])

    train_img_list = []
    train_model_list= []
    train_category_list = []
    for i in range(len(final_train_category_list)):
        img_list,model_list, category_list = resample(final_train_img_list[i],final_train_model_list[i], final_train_category_list[i], random_state=123,
                                                      n_samples= num_to_upsample if upsample else len(final_train_category_list[i]))
        train_img_list.append(img_list)
        train_model_list.append(model_list)
        train_category_list.append(category_list)

    return train_img_list,train_model_list,train_category_list

def get_train_test_histogram(root_path,json_path,train_file, test_file):
    y = json.load(open(json_path))
    train, test = get_train_test_indices(train_file, test_file)
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

def get_train_test_histogram_json(root_path,train_file, test_file, upsample):
    categories = json.load(open(train_file))['categories']
    category_list = [categories[i]['name'] for i in range(len(categories))]

    used_categories = category_list.copy()
    used_categories.remove('misc')
    used_categories.remove('tool')

    train_annotations = json.load(open(train_file))['annotations']
    test_annotations = json.load(open(test_file))['annotations']
    train_images = json.load(open(train_file))['images']
    test_images = json.load(open(test_file))['images']

    train_list = [0] * len(category_list)
    test_list = [0] * len(category_list)

    final_train_img_list = []
    final_train_model_list = []
    final_train_category_list = []

    for category in used_categories:
        train_img_list = []
        train_model_list= []
        train_category_list = []

        for i in range(len(train_annotations)):
            if category_list[train_annotations[i]['category_id']-1]==category:
                train_img_list.append(train_images[i]['img'])
                train_model_list.append(train_annotations[i]['voxel'])
                train_category_list.append(category_list[train_annotations[i]['category_id']-1]+'_pix3d')

        final_train_img_list.append(train_img_list)
        final_train_model_list.append(train_model_list)
        final_train_category_list.append(train_category_list)

    train_img_list,train_model_list,train_category_list = upsampling(final_train_img_list,final_train_model_list,final_train_category_list,upsample)
    # print(train_category_list)
    train_list = [len(cat_list) for cat_list in train_category_list]
    # print(train_list)

    for i in range(len(test_annotations)):
        # if not(category_list[test_annotations[i]['category_id']-1]=='misc' or category_list[test_annotations[i]['category_id']-1]=='tool'):
        test_list[test_annotations[i]['category_id']-1] += 1

    # del train_list[category_list.index('misc')]
    del test_list[category_list.index('misc')]
    del category_list[category_list.index('misc')]
    # del train_list[category_list.index('tool')]
    del test_list[category_list.index('tool')]

    xticks = [i for i in range(len(used_categories))]
    plt.bar(xticks, train_list, color='b', width=0.9)
    plt.bar(xticks, test_list, bottom=train_list, color='g', width=0.9)
    plt.xticks(xticks, used_categories)
    plt.title("Train-Test split of Classes in pix3D")
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

def get_train_test_splits(json_path,train_file, test_file):
    """
    :param filePath: pix3d.json path in dataset folder
    :return:
    """

    y = json.load(open(json_path))
    train, test = get_train_test_indices(train_file, test_file)
    # # print(y)
    # print(train)
    # print(test)
    train_img_list = []
    train_model_list= []
    train_category_list = []
    test_img_list = []
    test_model_list = []
    test_category_list = []

    for i in train:
        if not(y[i]['category']=='misc' or y[i]['category']=='tool'):
            train_img_list.append(y[i]['img'])
            train_model_list.append(y[i]['voxel'])
            train_category_list.append(y[i]['category'])

    for i in test:
        if not(y[i]['category']=='misc' or y[i]['category']=='tool'):
            test_img_list.append(y[i]['img'])
            test_model_list.append(y[i]['voxel'])
            test_category_list.append(y[i]['category'])

    return train_img_list,train_model_list,test_img_list,test_model_list,train_category_list,test_category_list

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

    # print(get_classes(root_path))
    # get_train_test_histogram(root_path,pix3d_json_path,os.path.join(os.path.dirname(__file__),'splits/pix3d_train.npy'),
    #                          os.path.join(os.path.dirname(__file__),'splits/pix3d_test.npy'))

##############################################################################################
    #voxel_path
    # voxel_path = get_voxel(root_path,"chair","0002.png")
    # voxel_path = '/Users/apple/OVGU/Thesis/SynthDataset1/models/chair/IKEA_HERMAN/voxel.mat'
    #
    # mesh_mat = load_mat_pix3d(voxel_path)
    # mesh_mat.show()

    #mode_path
    # model_path = get_model(root_path,"chair","0002.png")
    # mesh_mat = load(model_path)
    # mesh_mat.show()

##############################################################################################
    # train_img_list,train_model_list,test_img_list, \
    # test_model_list,train_category_list,test_category_list \
    #     = get_train_test_splits(pix3d_json_path,os.path.join(os.path.dirname(__file__),'splits/pix3d_train_2.npy'),
    #                                      os.path.join(os.path.dirname(__file__),'splits/pix3d_test_2.npy'))

    # test_model_list,train_model_list, train_category_list = resample(test_model_list,train_model_list, train_category_list, random_state=0)

##############################################################################################
    # get_model_names_histogram(pix3d_json_path)
    # # train_test_split_json(pix3d_json_path)
    # get_train_test_model_names_histogram(pix3d_json_path,os.path.join(os.path.dirname(__file__),'splits/pix3d_train_2.npy'),
    #                                      os.path.join(os.path.dirname(__file__),'splits/pix3d_test_2.npy'))

    get_train_test_model_names_histogram_json(pix3d_json_path,os.path.join(os.path.dirname(__file__),'splits/pix3d_s2_train.json'),
                                         os.path.join(os.path.dirname(__file__),'splits/pix3d_s2_test.json'))

    get_train_test_histogram_json(pix3d_json_path,os.path.join(os.path.dirname(__file__),'splits/pix3d_s2_train.json'),
                                  os.path.join(os.path.dirname(__file__),'splits/pix3d_s2_test.json'), upsample=False)