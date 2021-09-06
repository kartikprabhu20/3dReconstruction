"""

    Created on 01/06/21 11:14 AM
    @author: Kartik Prabhu

"""
import Baseconfig
from Datasets.MixedPix3DDataset import MixedPix3DDataset
from Datasets.pix3d.Pix3dDataset import Pix3dDataset
from Datasets.pix3dsynthetic.BaseDataset import BaseDataset
from Datasets.pix3dsynthetic.SyntheticPix3dDataset import SyntheticPix3dDataset
from Datasets.EmptyDataset import EmptyDataset
from Datasets.synth2real.Synth2realDataset import Synth2RealDataset


class DatasetType:
    PIX3D = "pix3d"
    PIX3DSYNTHETIC1 = "pix3dsynthetic1"
    EMPTY = "empty"
    SYNTH2REAL = "synth2real"
    MIXEDPIX3D = "mixedpix3d"

class LabelType:
    MAT = "mat"
    NPY = "npy"

class DatasetManager():
    def __init__(self,config):
        self.config = config

    def get_dataset(self, dataset_type, logger=None) -> BaseDataset:
        if DatasetType.PIX3DSYNTHETIC1 == dataset_type:
            return SyntheticPix3dDataset(config=self.config,logger=logger)
        elif DatasetType.PIX3D == dataset_type:
            return Pix3dDataset(config=self.config)
        elif DatasetType.SYNTH2REAL == dataset_type:
            return Synth2RealDataset(config=self.config)
        elif DatasetType.MIXEDPIX3D == dataset_type:
            return MixedPix3DDataset(config=self.config)
        elif DatasetType.EMPTY == dataset_type:
            return EmptyDataset(config=self.config)

        return SyntheticPix3dDataset(config=self.config,logger=logger)

if __name__ == '__main__':
    switcher = {
        DatasetType.PIX3D: "test",
        DatasetType.PIX3DSYNTHETIC1: "train"
    }
    print(switcher.get(Baseconfig.config.dataset_type, "false"))

    dataset = DatasetManager(Baseconfig.config).get_dataset(Baseconfig.config.dataset_type)
    print(dataset)

