"""

    Created on 08/01/21 4:00 PM
    @author: Kartik Prabhu

"""
import numpy as np
import scipy.io as sio
import trimesh
import os

def load_mat(filename, key):
    '''
    For file types .mat
    Converts to trimesh
    :param filename:
    :return:
    '''
    matstruct_contents = sio.loadmat(filename)
    mesh_mat = matstruct_contents[key][0]
    mesh_mat = trimesh.voxel.VoxelGrid(mesh_mat)
    return mesh_mat

def load(filename):
    '''
    For file types .off, .obj, .binvox
    :param filename: path to the file
    :return: trimesh object
    '''
    mesh = trimesh.load(filename)
    return mesh

if __name__ == '__main__':

    datapath_obj = '/Users/apple/OVGU/Thesis/Dataset/pix3d/model/bed/IKEA_BEDDINGE/model.obj'
    datapath_mat = '/Users/apple/OVGU/Thesis/Dataset/pix3d/model/bed/IKEA_BEDDINGE/voxel.mat'
    datapath_mat2 = '/Users/apple/OVGU/Thesis/Dataset/ShapeNet/val_voxels/000081/model.mat'

    datapath_off = '/Users/apple/OVGU/Thesis/Dataset/ModelNet40_small/bed/test/bed_0516.off'
    datapath_binvox = '/Users/apple/OVGU/Thesis/Dataset/ShapeNetVox32/02691156/1a04e3eab45ca15dd86060f189eb133/model.binvox'

    # mesh_obj = load(datapath_obj)
    # # mesh_obj.show()
    # print(mesh_obj)

    # mesh_mat = load_mat(datapath_mat, 'voxel')
    # mesh_mat.show()
    # print(mesh_mat)

    # mesh_mat = load_mat(datapath_mat2, 'input')
    # mesh_mat.show()
    # print(mesh_mat)

    # mesh_off = load(datapath_off)
    # # mesh_off.show()
    # print(mesh_off)
    #
    # mesh_binvox = load(datapath_binvox)
    # # mesh_binvox.show()
    # print(mesh_binvox)





