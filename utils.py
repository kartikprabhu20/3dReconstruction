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

def write_summary(writer, logger, index, original, reconstructed, focalTverskyLoss, diceLoss, diceScore, iou):
    """
    Method to write summary to the tensorboard.
    index: global_index for the visualisation
    original,reconstructer: cubified voxels [channel, Height, Width]
    Losses: all losses used as metric
    """
    print('Writing Summary...')
    writer.add_scalar('FocalTverskyLoss', focalTverskyLoss, index)
    writer.add_scalar('DiceLoss', diceLoss, index)
    writer.add_scalar('DiceScore', diceScore, index)
    writer.add_scalar('IOU', iou, index)

    writer.add_mesh('label', vertices=original.verts_list()[0][None,], faces=original.faces_list()[0][None,])
    writer.add_mesh('reconstructed', vertices=reconstructed.verts_list()[0][None,], faces=reconstructed.faces_list()[0][None,])
    # writer.add_image('diff', np.moveaxis(create_diff_mask(reconstructed,original,logger), -1, 0), index) #create_diff_mask is of the format HXWXC, but CXHXW is needed


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

def save_model(CHECKPOINT_PATH, state, best_metric = False,filename='checkpoint'):
    """
    Method to save model
    """
    print('Saving model...')
    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)
        if best_metric:
            if not os.path.exists(CHECKPOINT_PATH + 'best_metric/'):
                CHECKPOINT_PATH = CHECKPOINT_PATH + 'best_metric/'
                os.mkdir(CHECKPOINT_PATH)
    torch.save(state, CHECKPOINT_PATH + filename + str(state['epoch']) + '.pth')


if __name__ == '__main__':

    datapath_obj = '/Users/apple/OVGU/Thesis/Dataset/pix3d/model/bed/IKEA_BEDDINGE/model.obj'
    datapath_mat = '/Users/apple/OVGU/Thesis/SynthDataset1/models/wardrobe/IKEA_PAX_2/voxel.mat'
    datapath_mat2 = '/Users/apple/OVGU/Thesis/Dataset/ShapeNet/val_voxels/000081/model.mat'

    datapath_off = '/Users/apple/OVGU/Thesis/Dataset/ModelNet40_small/bed/test/bed_0516.off'
    datapath_binvox = '/Users/apple/OVGU/Thesis/Dataset/ShapeNetVox32/02691156/1a04e3eab45ca15dd86060f189eb133/model.binvox'

    # front = '/Users/apple/OVGU/Thesis/Dataset/3D-FUTURE-model_editedSmall/3D-FUTURE-model/0a5a346c-cc3b-4280-b358-ccd1c4d8a865/normalized_model.obj'
    # mesh_obj = load(front)
    # mesh_obj.show()
    # print(mesh_obj)

    mesh_mat = load_mat_pix3d(datapath_mat)
    # mesh_mat.show()

    writer_training = SummaryWriter("/Users/apple/OVGU/Thesis/code/3dReconstruction/outputs/test/")
    LOGGER_PATH = "/Users/apple/OVGU/Thesis/code/3dReconstruction/outputs/" + "MODEL_NAME"+'.log'
    logger = Logger("MODEL_NAME", LOGGER_PATH).get_logger()

    # mesh_mat = torch.tensor(mat_to_array(datapath_mat))[None,]
    mesh_mat = torch.ones(128,128,128)[None,]
    print(mesh_mat.shape)
    mesh_mat = cubify(mesh_mat, thresh=0.5)

    print(type(mesh_mat))
    save_obj("/Users/apple/OVGU/Thesis/code/3dReconstruction/outputs/test.obj",verts=mesh_mat.verts_list()[0], faces=mesh_mat.faces_list()[0])
    write_summary(writer_training, logger, 0, mesh_mat,
                  mesh_mat,
                  # 6 because in some cases we have padded with 5 which returns background
                  1, 1, 0, 0)

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
    # # mesh_binvox.show()
    # print(mesh_binvox)





