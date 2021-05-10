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
    datapath_np = '/Users/apple/OVGU/Thesis/code/3dReconstruction/80_val_output.npy'

    vis = Visualise(datapath_np)
    mesh = vis.np_to_mesh()
    save_obj("/Users/apple/OVGU/Thesis/code/3dReconstruction/80_val_output.obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])


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
