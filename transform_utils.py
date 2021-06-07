"""

    Created on 03/06/21 7:45 PM 
    @author: Kartik Prabhu

"""

import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))
        array = np.transpose(rendering_images, (0, 3, 1, 2))
        # handle numpy array
        tensor = torch.from_numpy(array)

        # put it from HWC to CHW format
        return tensor.float()


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        assert (isinstance(images, np.ndarray))
        images -= self.mean
        images /= self.std

        return images