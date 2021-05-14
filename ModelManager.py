"""

    Created on 13/05/21 11:31 PM 
    @author: Kartik Prabhu

"""
from models.pix2voxel import pix2vox


class ModelManager:
    PIX2VOXEL = "pix2voxel"

    def __init__(self, config):
        self.config = config


    def get_Model(self, model_type):
        switcher = {
            ModelManager.PIX2VOXEL: pix2vox(self.config),
        }
        return switcher.get(model_type, pix2vox(self.config))