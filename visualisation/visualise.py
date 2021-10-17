"""

    Created on 25/12/20 6:34 PM 
    @author: Kartik Prabhu

"""
import torch
import trimesh
import os
import numpy as np
from pytorch3d.ops import cubify
from skimage.filters import threshold_otsu
from torchviz import make_dot
from trimesh import Trimesh
from trimesh.exchange import binvox
from trimesh.exchange.binvox import load_binvox, Binvoxer
from trimesh.voxel.creation import voxelize_binvox

from utils import save_obj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


class Visualise():
    def __init__(self, file_path):
        self.file_path = file_path

    def __mesh__(self):
        mesh = trimesh.load(os.path.join(self.file_path, "bed/train/bed_0001.off"))
        # mesh.show()
        return mesh

    def np_to_mesh(self):
        mesh_np = np.load(self.file_path)
        return self.numpy_to_mesh(mesh_np)

    def numpy_to_mesh(self, numpyarray):
        thresh = threshold_otsu(numpyarray)
        mesh_np = (numpyarray > thresh).astype(int)
        mesh_mat = torch.tensor(mesh_np)[None,]
        mesh_mat = cubify(mesh_mat, thresh=0.5)
        return mesh_mat

    def binvox_to_mesh(self):
        mesh = trimesh.load(self.file_path)
        thresh = threshold_otsu(mesh)
        print(thresh)
        mesh_np = (mesh > thresh).astype(int)
        print(mesh_np)
        mesh_mat = torch.tensor(mesh_np)[None,]
        mesh_mat = cubify(mesh_mat, thresh=0.5)
        return mesh_mat

class Visualise_model():
    def __init__(self, model):
        self.model = model

    def visualise(self, input, output):
        output = self.model(input)
        make_dot(output)

def get_volume_views(volume, save_dir, n_itr):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    save_path = os.path.join(save_dir, 'voxels-%06d.png' % n_itr)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return cv2.imread(save_path)

def get_voxel_visualisation(voxels):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels, facecolors='gray' , edgecolor='k')

    plt.show()

def voxelize_binvox(mesh,model_path, dimension=None, **binvoxer_kwargs):

    from trimesh.exchange import binvox
    binvoxer = binvox.Binvoxer(dimension=dimension, **binvoxer_kwargs)
    return voxelize_mesh(mesh,model_path, binvoxer)

def voxelize_mesh(mesh,
                  model_path,
                  binvoxer=None):
    out_path = binvoxer(model_path,overwrite=True)
    with open(out_path, 'rb') as fp:
        out_model = load_binvox(fp)
    return out_model

if __name__ == '__main__':
    # model_name = "test_18"
    # epoch ="399"
    # path = '/Users/apple/OVGU/Thesis/code/3dReconstruction/'
    # # path ='/Users/apple/OVGU/Thesis/code/3dReconstruction/outputs/test_18/'
    # datapath_np = path+model_name+'_'+epoch+'_val_output.npy'
    #
    # vis = Visualise(datapath_np)
    # mesh = vis.np_to_mesh()
    # save_obj(path+model_name+'_'+epoch+"_val_output.obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
    #
    # datapath_np = path+model_name+'_'+epoch+'_val_original.npy'
    #
    # vis = Visualise(datapath_np)
    # mesh = vis.np_to_mesh()
    # save_obj(path+model_name+'_'+epoch+"_val_original.obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])


#########################################################
    # model = pix2vox(in_channel=config.in_channels)
    # train_img_list,train_model_list,test_img_list,test_model_list = get_train_test_split( config.root_path+"/pix3dsynthetic/train_test_split.p")
    # traindataset = SyntheticPix3d(config,train_img_list,train_model_list)
    # train_loader = torch.utils.data.DataLoader(traindataset, batch_size=config.batch_size, shuffle=True,
    #                                            num_workers=config.num_workers)
    # print(model)
    # for batch_index, (local_batch, local_labels) in enumerate(train_loader):
    #
    #     print(local_batch.shape)
    #     output = model(local_batch)
    #     make_dot(output, params=dict(list(model.named_parameters()))).render("model", format="png")
    #     break
    #
    # print("done")
#########################################################

#     model_name = "test_11"
#     epoch ="380"
#     datapath_np_output = '/Users/apple/OVGU/Thesis/code/3dReconstruction/'+model_name+'_'+epoch+'_val_output.npy'
#     datapath_np_original = '/Users/apple/OVGU/Thesis/code/3dReconstruction/'+model_name+'_'+epoch+'_val_original.npy'
#
#     #
#     #
#     predicted = np.load(datapath_np_output)
#     label = np.load(datapath_np_original)
#
#     thresh = threshold_otsu(predicted)
#     label =  label >thresh
#     predicted_binary = predicted > thresh
#
#     print(label)
#     print(predicted_binary)
#     # #
#     # # false_negative = np.subtract(label, predicted_binary) > 0
#     # # false_positive =  np.subtract(predicted_binary, label)>0
#     # # # # # combine the objects into a single boolean array
#     # voxels = predicted_binary | label
#     # # # # set the colors of each object
#     # colors = np.empty(label.shape, dtype=object)
#     # colors[label] = 'red'
#     # # # colors[false_positive] = 'blue'
#     # # # colors[predicted_binary] = 'green'
#     # # # colors[cube_label] = 'green'
#     # #
#     # # # and plot everything
#     # ax = plt.figure().add_subplot(projection='3d')
#     # ax.voxels(label, facecolors=colors, edgecolor='k')
#     # print("done")
#
# #######################################################################################################
#     x, y, z = np.indices((128, 128,128))
#
#     # draw cuboids in the top left and bottom right corners, and a link between
#     # them
#     cube1 = (x < 3) & (y < 3) & (z < 3)
#     cube2 = (x >= 5) & (y >= 5) & (z >= 5)
#     link = abs(x - y) + abs(y - z) + abs(z - x) <= 20
#
#     print(cube1)
#     # combine the objects into a single boolean array
#     voxels = cube1 | cube2 | link
#
#     print(voxels)
#
#     # set the colors of each object
#     colors = np.empty(voxels.shape, dtype=object)
#     colors[link] = 'red'
#     colors[cube1] = 'blue'
#     colors[cube2] = 'green'
#
#     print(colors)
#
#     # and plot everything
#     ax = plt.figure().add_subplot(projection='3d')
#     ax.voxels(voxels, facecolors=colors, edgecolor='k')
#
#     plt.show()
########################################################
    datapath_np = "/Users/apple/OVGU/Thesis/Dataset/pix3d/model/chair/IKEA_BERNHARD/voxel_128.npy"

    vis = Visualise(datapath_np)
    mesh = vis.np_to_mesh()
    save_obj("/Users/apple/OVGU/Thesis/Dataset/pix3d/model/chair/IKEA_BERNHARD/voxel_128.obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])

    datapath_np = "/Users/apple/OVGU/Thesis/Dataset/pix3d/model/chair/IKEA_BERNHARD/model.binvox"

    vis = Visualise(datapath_np)
    mesh = vis.binvox_to_mesh()
    save_obj("/Users/apple/OVGU/Thesis/Dataset/pix3d/model/chair/IKEA_BERNHARD/voxel_128.obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])

    # model_folder ='/Users/apple/OVGU/Thesis/Dataset/pix3d/model/desk/IKEA_GALANT_2/'
    # datapath = model_folder+'model.obj'
    # binvox_path = '/Users/apple/OVGU/Thesis/binvox'
    # datapath_binvox = model_folder+'model.binvox'
    # dim = 32
    #
    # trimesh = trimesh.load(datapath, force='mesh')
    # binvoxObj = voxelize_binvox(trimesh,model_path=datapath,dimension=dim,binvox_path=binvox_path,dilated_carving=True,downsample_threshold=128)
    #
    # vis = Visualise(datapath_binvox)
    # mesh = vis.numpy_to_mesh(binvoxObj.matrix)
    # save_obj(model_folder+"voxel_"+str(dim)+".obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
    #
    # import binvox_rw
    # with open(datapath_binvox, 'rb') as f:
    #     model = binvox_rw.read_as_3d_array(f)

########################################################
    #dilation
    # import scipy.ndimage
    # scipy.ndimage.binary_dilation(model.data.copy(), output=model.data)
    #
    # mesh = vis.numpy_to_mesh(model.data)
    # save_obj(model_folder+"voxel_binvox_dilated_"+str(dim)+".obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
    #
    # model_npy = np.load(model_folder+"voxel_"+str(dim)+".npy")
    # model_npy = model_npy > 0
    # mesh = vis.numpy_to_mesh(model_npy)
    # save_obj(model_folder+"voxel_npy_"+str(dim)+".obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])
    #
    # model_npy_copy = model_npy
    # scipy.ndimage.binary_dilation(model_npy_copy, output=model_npy)
    # mesh = vis.numpy_to_mesh(model_npy)
    # save_obj(model_folder+"voxel_npy_dilated_"+str(dim)+".obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])

########################################################

