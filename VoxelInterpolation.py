
"""
    Created on 08/01/21 4:00 PM
    @author: Kartik Prabhu
    Reference: https://github.com/xingyuansun/pix3d/blob/master/eval/eval.py
"""

import numpy as np
import numba
from pytorch3d.ops import cubify
from scipy.interpolate import RegularGridInterpolator as rgi
import scipy.io as sio
import PIL
import torch
from skimage.filters import threshold_otsu
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import io
import os
from utils import save_obj

def interp3(V, xi, yi, zi, fill_value=0):
    x = np.arange(V.shape[0])
    y = np.arange(V.shape[1])
    z = np.arange(V.shape[2])
    interp_func = rgi((x, y, z), V, 'linear', False, fill_value)
    return interp_func(np.array([xi, yi, zi]).T)


def mesh_grid(input_lr, output_size):
    x_min, x_max, y_min, y_max, z_min, z_max = input_lr
    length = max(max(x_max - x_min, y_max - y_min), z_max - z_min)
    center = np.array([x_max - x_min, y_max - y_min, z_max - z_min]) / 2.
    x = np.linspace(center[0] - length / 2, center[0] + length / 2, output_size[0])
    y = np.linspace(center[1] - length / 2, center[1] + length / 2, output_size[1])
    z = np.linspace(center[2] - length / 2, center[2] + length / 2, output_size[2])
    return np.meshgrid(x, y, z)


def thresholding(V, threshold):
    """
    return the original voxel in its bounding box and bounding box coordinates.
    """
    if V.max() < threshold:
        return np.zeros((2,2,2)), 0, 1, 0, 1, 0, 1
    V_bin = (V >= threshold)
    x_sum = np.sum(np.sum(V_bin, axis=2), axis=1)
    y_sum = np.sum(np.sum(V_bin, axis=2), axis=0)
    z_sum = np.sum(np.sum(V_bin, axis=1), axis=0)

    x_min = x_sum.nonzero()[0].min()
    y_min = y_sum.nonzero()[0].min()
    z_min = z_sum.nonzero()[0].min()
    x_max = x_sum.nonzero()[0].max()
    y_max = y_sum.nonzero()[0].max()
    z_max = z_sum.nonzero()[0].max()
    return V[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1], x_min, x_max, y_min, y_max, z_min, z_max

downsample_uneven_warned = False


def downsample(vox_in, times, use_max=True):
    global downsample_uneven_warned
    if vox_in.shape[0] % times != 0 and not downsample_uneven_warned:
        print('WARNING: not dividing the space evenly.')
        downsample_uneven_warned = True
    return _downsample(vox_in, times, use_max=use_max)


@numba.jit(nopython=True, cache=True)
def _downsample(vox_in, times, use_max=True):
    dim = vox_in.shape[0] // times
    vox_out = np.zeros((dim, dim, dim))
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                subx = x * times
                suby = y * times
                subz = z * times
                subvox = vox_in[subx:subx + times,
                         suby:suby + times, subz:subz + times]
                if use_max:
                    vox_out[x, y, z] = np.max(subvox)
                else:
                    vox_out[x, y, z] = np.mean(subvox)
    return vox_out


def downsample_voxel(voxel, threshold, output_size, resample=True):
    if voxel.shape[0] > 100:
        assert output_size[0] in (32,64, 128)
        # downsample to 32 before finding bounding box
        if output_size[0] == 32:
            voxel = downsample(voxel, 4, use_max=True)
        if output_size[0] == 64:
            voxel = downsample(voxel, 2, use_max=True)
    if not resample:
        return voxel

    voxel, x_min, x_max, y_min, y_max, z_min, z_max = thresholding(
        voxel, threshold)
    x_mesh, y_mesh, z_mesh = mesh_grid(
        (x_min, x_max, y_min, y_max, z_min, z_max), output_size)
    x_mesh = np.reshape(np.transpose(x_mesh, (1, 0, 2)), (-1))
    y_mesh = np.reshape(np.transpose(y_mesh, (1, 0, 2)), (-1))
    z_mesh = np.reshape(z_mesh, (-1))

    fill_value = 0
    voxel_d = np.reshape(interp3(voxel, x_mesh, y_mesh, z_mesh, fill_value),
                         (output_size[0], output_size[1], output_size[2]))
    return voxel_d

def mat_to_array(filename,key='voxel'):
    matstruct_contents = sio.loadmat(filename)
    mesh_mat = matstruct_contents[key]
    return np.array(mesh_mat)


def gen_plot(voxels, savefig = False, path = None):
    """Create a pyplot plot and save to buffer."""
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels,facecolors='gray',edgecolor='k')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(path)
    buf.seek(0)
    plt.close('all')
    return buf

def plot_to_tensor(plot_buf):
    image = PIL.Image.open(plot_buf)
    return  ToTensor()(image)

def numpy_to_mesh(numpyarray):
    thresh = threshold_otsu(numpyarray)
    mesh_np = (numpyarray > thresh).astype(int)
    mesh_mat = torch.tensor(mesh_np)[None,]
    mesh_mat = cubify(mesh_mat, thresh=0.5)
    return mesh_mat

def convert(filename,outpath,threshold = 0.1, size=(64,64,64), key='voxel'):
    data = mat_to_array(filename,key=key)
    # data = torch.tensor(data)
    output = downsample_voxel(data,threshold,size)
    voxel_path = os.path.join(outpath,"voxel_"+str(size[0]))
    np.save(voxel_path, output)

    model_npy = np.load(voxel_path+".npy")
    model_npy = model_npy > 0
    gen_plot(model_npy, savefig=True, path = os.path.join(outpath,"voxel_npy_"+str(size[0])+".png"))

    # if size[0]==32:
    #   mesh = numpy_to_mesh(model_npy)
#       save_obj( os.path.join(outpath,"voxel_npy_"+str(size[0])+".obj"),verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])

if __name__ == '__main__':
    datapath_mat = '/Users/apple/OVGU/Thesis/Dataset/pix3d/model/chair/IKEA_BORJE/voxel.mat'
    outpath = '/Users/apple/OVGU/Thesis/'
    dim = 64
    thresh = 0.2
    convert(datapath_mat,outpath,thresh, size=(dim,dim,dim))

    model_npy = np.load(outpath+"voxel_"+str(dim)+".npy")
    model_npy = model_npy > 0
    mesh = numpy_to_mesh(model_npy)
    gen_plot(model_npy, savefig=True, path = outpath + "voxel" + str(dim) + ".png")
    save_obj(outpath+"voxel_npy_"+str(dim)+".obj",verts=mesh.verts_list()[0], faces=mesh.faces_list()[0])

