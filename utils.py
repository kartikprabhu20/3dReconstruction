"""

    Created on 08/01/21 4:00 PM
    @author: Kartik Prabhu

"""
import numpy as np
import scipy.io as sio
import trimesh
import os

def load_obj(filename):
    """Load vertices from .obj wavefront format file."""
    # vertices = []
    # with open(filename, 'r') as mesh:
    #     for line in mesh:
    #         data = line.split()
    #         if len(data) > 0 and data[0] == 'v':
    #             vertices.append(data[1:])
    mesh = trimesh.load(filename)
    return mesh

# def load_mat(filename):
#     matstruct_contents = sio.loadmat(filename)
#     return matstruct_contents
#
# def load_off(filename):
#     mesh = trimesh.exchange.off.load_off(filename)
#     return mesh
#
# def load_binvox(filename):
#     mesh = trimesh.exchange.binvox.load_binvox(filename)
#     return mesh


def load(filename):
    '''
    For file types .off, .obj, .binvox
    :param filename:
    :return:
    '''
    mesh = trimesh.load(filename)
    return mesh

if __name__ == '__main__':

    datapath_obj = '/Users/apple/OVGU/Thesis/Dataset/pix3d/model/bed/IKEA_BEDDINGE/'
    datapath_mat = '/Users/apple/OVGU/Thesis/Dataset/pix3d/model/bed/IKEA_BEDDINGE/voxel.mat'
    datapath_off = '/Users/apple/OVGU/Thesis/Dataset/ModelNet40_small/bed/test/bed_0516.off'
    datapath_binvox = '/Users/apple/OVGU/Thesis/Dataset/ShapeNetVox32/02691156/1a04e3eab45ca15dd86060f189eb133/model.binvox'

    mesh_obj = load(os.path.join(datapath_obj, 'model.obj'))
    # mesh_obj.show()
    print(mesh_obj)

    mesh_mat = load_mat(datapath_mat)
    mesh_mat = mesh_mat['voxel']
    print(mesh_mat)

    mesh_off = load(datapath_off)
    # mesh_off.show()
    print(mesh_off)

    mesh_binvox = load(datapath_binvox)
    mesh_binvox.show()
    print(mesh_binvox)





