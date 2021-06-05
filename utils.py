"""

    Created on 08/01/21 4:00 PM
    @author: Kartik Prabhu

"""
import numpy as np
import scipy.io as sio
import torch
import trimesh
import os

from iopath import PathManager
from pytorch3d.ops import cubify
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from Logging import Logger
from pytorch3d.io.utils import _open_file
from skimage.filters import threshold_otsu
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def create_binary(predicted, logger):

    predicted = predicted.cpu().data.numpy()

    try:
        thresh = threshold_otsu(predicted)
        predicted_binary = (predicted > thresh).astype(int)
    except Exception as error:
        logger.exception(error)
        predicted_binary = (predicted > 0.5).astype(int)  # exception will be thrown only if input image seems to have just one color 1.0.

    return torch.tensor(predicted_binary, dtype=torch.uint8)

def load_mat_shapenet(filename, key='input'):
    '''
    For file types .mat
    Converts to trimesh
    :param filename:
    :return:
    '''
    matstruct_contents = sio.loadmat(filename)
    mesh_mat = matstruct_contents[key][0]
    return trimesh.voxel.VoxelGrid(mesh_mat)

def load_mat_pix3d(filename, key='voxel'):
    '''
    For file types .mat
    Converts to trimesh
    :param filename:
    :return:
    '''
    matstruct_contents = sio.loadmat(filename)
    print(matstruct_contents)
    mesh_mat = matstruct_contents[key]
    mesh_mat = torch.tensor(mesh_mat)
    mesh_mat = mesh_mat.numpy()
    return trimesh.voxel.VoxelGrid(mesh_mat)

def save_mat_to_nparray(filename,outfile, key='voxel'):
    data = mat_to_array(filename,key='voxel')
    np.save(outfile, data)

def mat_to_array(filename,key='voxel'):
    matstruct_contents = sio.loadmat(filename)
    mesh_mat = matstruct_contents[key]
    return np.array(mesh_mat)

def load(filename):
    '''
    For file types .off, .obj, .binvox
    :param filename: path to the file
    :return: trimesh object
    '''
    mesh = trimesh.load(filename)
    return mesh

def save_obj(
        f,
        verts,
        faces,
        decimal_places: Optional[int] = None,
        path_manager: Optional[PathManager] = None,
):
    """
    Save a mesh to an .obj file.
    Args:
        f: File (or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        decimal_places: Number of decimal places for saving.
        path_manager: Optional PathManager for interpreting f if
            it is a str.
    """
    if len(verts) and not (verts.dim() == 2 and verts.size(1) == 3):
        message = "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if len(faces) and not (faces.dim() == 2 and faces.size(1) == 3):
        message = "Argument 'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if path_manager is None:
        path_manager = PathManager()

    with _open_file(f, path_manager, "w") as f:
        return _save(f, verts, faces, decimal_places)


def _save(f, verts, faces,decimal_places) -> None:
    assert not len(verts) or (verts.dim() == 2 and verts.size(1) == 3)
    assert not len(faces) or (faces.dim() == 2 and faces.size(1) == 3)

    if not (len(verts) or len(faces)):
        print("Empty 'verts' and 'faces' arguments provided")
        return

    verts, faces = verts.cpu(), faces.cpu()

    lines = ""

    if len(verts):
        if decimal_places is None:
            float_str = "%f"
        else:
            float_str = "%" + ".%df" % decimal_places

        V, D = verts.shape
        for i in range(V):
            vert = [float_str % verts[i, j] for j in range(D)]
            lines += "v %s\n" % " ".join(vert)

    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        print("Faces have invalid indices")

    if len(faces):
        F, P = faces.shape
        for i in range(F):
            face = ["%d" % (faces[i, j] + 1) for j in range(P)]
            if i + 1 < F:
                lines += "f %s\n" % " ".join(face)
            elif i + 1 == F:
                # No newline at the end of the file.
                lines += "f %s" % " ".join(face)

    f.write(lines)

def interpolate_and_save(filename,outpath,size=64, key='voxel',scale_factor=1,mode='bilinear'):
    data = mat_to_array(filename,key=key)
    data = torch.tensor(data)
    # print(data.shape)
    data = data[None,None,]
    # print(data.shape)
    output = F.interpolate(input= data, size=(size,size,size),mode=mode)
    # print(output.shape)
    voxel_path = os.path.join(outpath,"voxel_"+str(size))
    np.save(voxel_path, output[0][0].numpy())



# mesh = np_to_mesh(outfile+"_"+str(size)+".npy")
    # save_obj(outfile+"_"+str(size)+".obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])


def np_to_mesh(file_path):
    mesh_np = np.load(file_path)
    print(mesh_np.shape)
    mesh_mat = torch.tensor(mesh_np)[None,]
    mesh_mat = cubify(mesh_mat, thresh=0.5)
    return mesh_mat

def save_volume_views(volume, save_dir, index):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    save_path = os.path.join(save_dir, 'voxels' + index + '.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':

    datapath_obj = '/Users/apple/OVGU/Thesis/Dataset/pix3d/model/bed/IKEA_BEDDINGE/model.obj'
    datapath_mat = '/Users/apple/OVGU/Thesis/Dataset/pix3d/model/chair/IKEA_HERMAN/voxel.mat'
    datapath_mat2 = '/Users/apple/OVGU/Thesis/Dataset/ShapeNet/val_voxels/000081/model.mat'

    datapath_off = '/Users/apple/OVGU/Thesis/Dataset/ModelNet40_small/bed/test/bed_0516.off'
    datapath_binvox = '/Users/apple/OVGU/Thesis/SynthDataset1/models/chair/IKEA_HERMAN/model_2.binvox'
    # front = '/Users/apple/OVGU/Thesis/Dataset/3D-FUTURE-model_editedSmall/3D-FUTURE-model/0a5a346c-cc3b-4280-b358-ccd1c4d8a865/normalized_model.obj'

    ##############################################################################


    # save_mat_to_nparray(datapath_mat,"/Users/apple/Desktop/datapath_mat")
    # print(torch.tensor(mat_to_array(datapath_mat)))

    # mesh_mat = load_mat_shapenet(datapath_mat2)
    # mesh_mat.show()
    # print(mesh_mat)

    # mesh_off = load(datapath_off)
    # # mesh_off.show()
    # print(mesh_off)
    #
    # mesh_binvox = load(datapath_binvox)
    # mesh_binvox.show()
    # print(mesh_binvox)

    ##############################################################################
    # interpolate_and_save(datapath_mat,"/Users/apple/Desktop/datapath_mat2",size=32, mode='nearest')
    # interpolate_and_save(datapath_mat,"/Users/apple/Desktop/datapath_mat2",size=64,mode='nearest')
    # interpolate_and_save(datapath_mat,"/Users/apple/Desktop/datapath_mat2",size=128,mode='nearest')
    ##############################################################################

