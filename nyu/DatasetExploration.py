"""

    Created on 16/02/21 10:37 AM 
    @author: Kartik Prabhu

"""

import skimage.io as io
import numpy as np
import h5py
from PIL import Image
import os
import torch
try:
    import accimage
except ImportError:
    accimage = None
from torchvision import transforms
import matplotlib.pyplot as plt


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, image):
        image = self.to_tensor(image)
        return image


    def to_tensor(self,pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):

            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)


        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def colored_depthmap(depth, d_min=None, d_max=None):
    # cmap = plt.cm.viridis
    cmap = plt.get_cmap('jet')
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def img_transform(img):
    """Normalize an image."""

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = data_transform(img)
    return img

def nyu_np_files():
    rgb = np.load(os.path.join("/Users/apple/OVGU/Thesis/Dataset/nyu_test/", 'eigen_test_rgb.npy'))
    img = Image.fromarray(rgb[0].astype(np.uint8), mode = 'RGB')
    img.show()

    depth = np.load(os.path.join("/Users/apple/OVGU/Thesis/Dataset/nyu_test/", 'eigen_test_depth.npy'))
    depth = np.clip(np.load(os.path.join("/Users/apple/OVGU/Thesis/Dataset/nyu_test/", 'eigen_test_depth.npy')), 1.0, 10.0) / 10 * 255

    img_depth = Image.fromarray(depth[0].astype(np.uint8)[:,:], mode='L')
    img_depth.show()
    rgba_img = colored_depthmap(depth[0])
    print(rgba_img.shape)
    img_depth = Image.fromarray(rgba_img.astype(np.uint8))
    img_depth.show()

def nyu_mat_files():
    path_to_depth = '/Users/apple/OVGU/Thesis/Dataset/nyu_depth_v2_labeled.mat'

    # read mat file
    f = h5py.File(path_to_depth)

    # img = Image.fromarray(f['depths'][0].transpose(2,1,0))
    # img.show()

    # read 0-th image. original format is [3 x 640 x 480], uint8
    img = f['images'][0]

    # reshape
    img_ = np.empty([480, 640, 3])
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T

    # imshow
    img__ = img_.astype('float32')
    io.imshow(img__/255.0)
    io.show()


    # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
    depth = f['depths'][0]
    # reshape for imshow
    depth_ = np.empty([480, 640, 3])
    depth_[:,:,0] = depth[:,:].T
    depth_[:,:,1] = depth[:,:].T
    depth_[:,:,2] = depth[:,:].T
    io.imshow(depth_/4.0)
    io.show()

if __name__ == '__main__':
    # nyu_mat_files()

    # nyu_np_files()











