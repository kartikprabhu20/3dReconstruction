"""

    Created on 30/04/21 9:42 PM
    @author: Kartik Prabhu

"""
from easydict import EasyDict as edict
from sys import platform

from DatasetManager import LabelType, DatasetType
from ModelManager import OptimizerType, ModelType, LossTypes
from PipelineManager import PipelineType

cfg = edict()
config = cfg

cfg.main_name = 'test_25'
cfg.platform = platform
cfg.apex = False
cfg.apex_mode = "O2"

if platform == "darwin": #local mac
    cfg.root_path = '/Users/apple/OVGU/Thesis/code/3dReconstruction'
    cfg.dataset_path ='/Users/apple/OVGU/Thesis/SynthDataset1'
    cfg.output_path = "/Users/apple/OVGU/Thesis/code/3dReconstruction/outputs/"
    cfg.home_path = "/Users/apple/Desktop/"
    cfg.num_workers = 0
    cfg.batch_size = 8
    cfg.apex = False
else:
    cfg.root_path ='/nfs1/kprabhu/3dReconstruction1/'
    cfg.dataset_path ='/nfs1/kprabhu/SynthDataset1/'
    cfg.output_path = "/nfs1/kprabhu/outputs/"
    cfg.home_path = "/nfs1/kprabhu/"
    cfg.checkpoint_path = "/nfs1/kprabhu/outputs/" + cfg.main_name +"/"
    cfg.num_workers = 8
    cfg.batch_size = 24


cfg.tensorboard_train = cfg.output_path + cfg.main_name  + '/tensorboard/tensorboard_training/'
cfg.tensorboard_validation = cfg.output_path + cfg.main_name  + '/tensorboard/tensorboard_validation/'
cfg.checkpoint_path = cfg.output_path + cfg.main_name + "/"

cfg.num_epochs = 400
cfg.learning_rate = 0.0001

####### INPUT-OUTPUT CONFIG #######
cfg.include_depthmaps = False
cfg.include_masks = False
cfg.is_mesh = False
cfg.label_type = LabelType.NPY
cfg.voxel_size = 32 #32,64 or 128

cfg.in_channels = 3 #rgb
if cfg.include_depthmaps:
    cfg.in_channels += 1
if cfg.include_masks:
    cfg.in_channels += 1

cfg.resize = True
cfg.size = [224,224]
####### INPUT-OUTPUT CONFIG #######

cfg.train_loss_type = LossTypes.DICE
cfg.save_mesh = 1

########## Model Config################
cfg.pipeline_type = PipelineType.RECONSTRUCTION
cfg.dataset_type =  DatasetType.PIX3DSYNTHETIC1
cfg.model_type = ModelType.PIX2VOXEL
cfg.optimizer_type = OptimizerType.ADAM

cfg.pix2vox_pretrained_vgg = True
cfg.pix2vox_update_vgg = True
cfg.pix2vox_refiner = True

########## Model Config################


#
# Training
#
cfg.TRAIN                                   = edict()
cfg.TRAIN.BRIGHTNESS                        = .4
cfg.TRAIN.CONTRAST                          = .4
cfg.TRAIN.SATURATION                        = .4
cfg.TRAIN.NOISE_STD                         = .1
cfg.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
# cfg.TRAIN.ENCODER_LR_MILESTONES             = [150]
# cfg.TRAIN.DECODER_LR_MILESTONES             = [150]
# cfg.TRAIN.REFINER_LR_MILESTONES             = [150]
# cfg.TRAIN.MERGER_LR_MILESTONES              = [150]
cfg.TRAIN.BETAS                             = (.9, .999)
cfg.TRAIN.MOMENTUM                          = .9
cfg.TRAIN.GAMMA                             = .5
cfg.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
cfg.TRAIN.UPDATE_N_VIEWS_RENDERING          = False

