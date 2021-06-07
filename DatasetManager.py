"""

    Created on 01/06/21 11:14 AM
    @author: Kartik Prabhu

"""
from Datasets.pix3d.Pix3dDataset import Pix3dDataset
from Datasets.pix3dsynthetic.BaseDataset import BaseDataset
from Datasets.pix3dsynthetic.SyntheticPix3dDataset import SyntheticPix3dDataset


class DatasetType:
    PIX3D = "pix3d"
    PIX3DSYNTHETIC1 = "pix3dsynthetic1"

class LabelType:
    MAT = "mat"
    NPY = "npy"

class DatasetManager():
    def __init__(self,config):
        self.config = config

    def get_dataset(self, dataset_type) -> BaseDataset:

        switcher = {
            DatasetType.PIX3D: Pix3dDataset(self.config),
            DatasetType.PIX3DSYNTHETIC1: SyntheticPix3dDataset(config=self.config)
        }
        return switcher.get(dataset_type, SyntheticPix3dDataset(config=self.config))
