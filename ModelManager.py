"""

    Created on 13/05/21 11:31 PM 
    @author: Kartik Prabhu

"""
from models.pix2voxel import pix2vox
# Added for Apex
import apex
from apex import amp
import torch

class ModelType:
    PIX2VOXEL = "pix2voxel"

class OptimizerType:
    ADAM = "adam"

class LossTypes:
    BCE="bce"
    DICE="dice"
    FOCAL_TVERSKY="ftloss"
    IOU="iou"

class ModelManager:

    def __init__(self, config):
        self.config = config

    def get_Model(self, model_type):
        switcher = {
            ModelType.PIX2VOXEL: pix2vox(self.config),
        }

        self.model = switcher.get(model_type, pix2vox(self.config))
        if self.config.platform != "darwin": #no nvidia on mac!!!
            self.model.cuda()
        optimizer = self.get_optimiser(self.config.optimizer_type)

        if self.config.apex:
            amp.register_float_function(torch, 'sigmoid') #As the error message suggests, you could e.g. change the criterion to nn.BCEWithLogitsLoss and remove the sigmoid, register the sigmoid as a float function, or disable the warning.
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.config.apex_mode)

        return self.model,optimizer

    def get_optimiser(self, optimizer_type):

        switcher = {
        OptimizerType.ADAM: torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate),
        }
        return switcher.get(optimizer_type, torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate))

