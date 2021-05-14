"""

    Created on 25/12/20 6:34 PM 
    @author: Kartik Prabhu

"""
import torch
import trimesh
import glob
import os
import numpy as np
from pytorch3d.ops import cubify
from torchviz import make_dot

from Baseconfig import config
from models.pix2voxel import pix2vox
from pix3dsynthetic.DataExploration import get_train_test_split
from pix3dsynthetic.SyntheticPix3dDataset import SyntheticPix3d
from utils import save_obj



class Visualise():
    def __init__(self, file_path):
        self.file_path = file_path

    def __mesh__(self):
        mesh = trimesh.load(os.path.join(self.file_path, "bed/train/bed_0001.off"))
        # mesh.show()
        return mesh

    def np_to_mesh(self):
        mesh_np = np.load(self.file_path)
        mesh_mat = torch.tensor(mesh_np)[None,]
        mesh_mat = cubify(mesh_mat, thresh=0.5)
        return mesh_mat

class Visualise_model():
    def __init__(self, model):
        self.model = model

    def visualise(self, input, output):
        output = self.model(input)
        make_dot(output)

if __name__ == '__main__':
    model_name = "test_13"
    epoch ="240"
    path = '/Users/apple/OVGU/Thesis/code/3dReconstruction/outputs/test_13/'
    datapath_np = path+model_name+'_'+epoch+'_val_output.npy'

    vis = Visualise(datapath_np)
    mesh = vis.np_to_mesh()
    save_obj(path+model_name+'_'+epoch+"_val_output.obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])

    datapath_np = path+model_name+'_'+epoch+'_val_original.npy'

    vis = Visualise(datapath_np)
    mesh = vis.np_to_mesh()
    save_obj(path+model_name+'_'+epoch+"_val_original.obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])


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