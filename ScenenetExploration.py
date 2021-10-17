"""

    Created on 27/08/21 8:24 AM 
    @author: Kartik Prabhu

"""
import os

import trimesh
import matplotlib.pyplot as plt
import numpy as np

def getRooms(rootpath):
    rooms = []
    roomCategory = []

    for folder in os.listdir(rootpath):
        if folder =='.DS_Store':
            continue

        roomPath  = os.path.join(rootpath,folder)
        for filename in os.listdir(roomPath):
            rooms.append(os.path.join(roomPath,filename))
            roomCategory.append(folder)

    return rooms,roomCategory

def get_room_obj(filepath):
    mesh = trimesh.load(filepath)
    print(mesh)

    for obj in mesh.geometry:
        print(obj)
    print(mesh.geometry)

def get_histogram(fileName):
    plt.clf()
    names = ["LivingRoom","Kitchen","Bedroom", "Office"]
    quantity = [7,1,11,6]

    plt.bar(names, quantity)
    plt.title("Type of scenes")
    plt.tick_params(labelsize=14)
    # plt.xticks(xticks, model_names)
    # plt.legend(['Mode'])
    plt.savefig(fileName, bbox_inches='tight',format='pgf')

def get_histogram_2(fileName):
    plt.clf()
    names =["chair","bed","desk","bookcase","sofa","table","wardrobe"]
    quantity = [113,24,37,20,23,34,11,]

    plt.bar(names, quantity)
    plt.title("Pix3D Categories in SceneNet")
    plt.tick_params(labelsize=14)
    # plt.xticks(xticks, model_names)
    # plt.legend(['Mode'])
    plt.savefig(fileName, bbox_inches='tight',format='pgf')

def get_histogram_texture(texturepath,fileName):
    plt.clf()
    names =[]
    quantity = []

    c = 0
    for folder in os.listdir(texturepath):
        print(folder)
        if(folder == ".DS_Store"):
            continue
        names.append(folder)
        count = len(os.listdir(os.path.join(texturepath,folder)))
        quantity.append(count)
        c += count

    print(c)

    names = np.array(names)
    quantity = np.array(quantity)
    inds = quantity.argsort()
    sortedNames = names[inds][::-1][0:40]
    sortedquantity = quantity[inds][::-1][0:40]

    plt.bar(sortedNames, sortedquantity)
    plt.title("Distribution of textures")
    plt.xticks(rotation=90)
    plt.savefig(fileName, bbox_inches='tight',format='pgf')

if __name__ == '__main__':

    rootpath = "/Users/apple/OVGU/Thesis/scenenet/robotvault-downloadscenenet-cfe5ab85ddcc/3d-scene"
    rooms, roomcategory = getRooms(rootpath)
    texturepath = "/Users/apple/OVGU/Thesis/scenenet/robotvault-downloadscenenet-cfe5ab85ddcc/texture_library"
    # get_room_obj(rooms[0])
    get_histogram("/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/implementation/scenenet_scenes/types_of_scenes.pgf")
    get_histogram_2("/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/implementation/scenenet_scenes/distribution_categories_scenenet.pgf")
    get_histogram_texture(texturepath,"/Users/apple/OVGU/Thesis/code/3dReconstruction/report/images/implementation/scenenet_scenes/distribution_of_textures.pgf")