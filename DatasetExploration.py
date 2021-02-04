
"""

    Created on 05/01/21 6:34 PM
    @author: Kartik Prabhu

"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json
import os

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
    #return ["bed","bookcase","chair","desk","sofa","table"]
    return [class_folder for class_folder in os.listdir(root_path+"/img/") if not class_folder.startswith('.')]

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



if __name__ == '__main__':

    pix3d_json_path = '/Users/apple/OVGU/Thesis/Dataset/pix3d/pix3d.json'
    root_path = "/Users/apple/OVGU/Thesis/Dataset/pix3d/"

    y = json.load(open(pix3d_json_path))
    print(type(y))
    print(type(y[2]))

    #split the json into class
    # split_to_classes(pix3d_json_path,root_path)

    ##Histogram test
    # getHistogram("/Users/apple/OVGU/Thesis/Dataset/pix3d/img/")

    #Save occluded json
    # save_occluded_json(root_path)

    #Occluded Histogram
    # get_occluded_Histogram(root_path)



