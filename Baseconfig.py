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

cfg.main_name = 'test_57'
cfg.pretrained_name = 'test_57'
cfg.platform = platform
cfg.apex = False
cfg.apex_mode = "O2"
cfg.load_model = False #For direct testing

cfg.num_epochs = 800
cfg.learning_rate = 0.001

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
cfg.crop_size = [128,128]

####### INPUT-OUTPUT CONFIG #######

cfg.train_loss_type = LossTypes.BCE
cfg.save_mesh = 20
cfg.save_count = 4 # <batch_size

########## Model Config################

# Dataset
#
cfg.DATASET                                 = edict()
cfg.DATASET.MEAN                            = [0.485, 0.456, 0.406]
cfg.DATASET.STD                             = [0.229, 0.224, 0.225]

cfg.pipeline_type = PipelineType.RECONSTRUCTION
cfg.dataset_type = DatasetType.PIX3DSYNTHETIC1
cfg.model_type = ModelType.PIX2VOXELPP
cfg.optimizer_type = OptimizerType.ADAM

cfg.pix2vox_pretrained_encoder = True
cfg.pix2vox_update_vgg = True
cfg.pix2vox_refiner = True
cfg.pix2vox_merger = True

########## Model Config################


#
# Training
#
cfg.TRAIN                                   = edict()
cfg.TRAIN.RESUME_TRAIN                      = False
cfg.TRAIN.NUM_WORKER                        = 4             # number of data workers
cfg.TRAIN.NUM_EPOCHES                       = 250
cfg.TRAIN.BRIGHTNESS                        = .4
cfg.TRAIN.CONTRAST                          = .4
cfg.TRAIN.SATURATION                        = .4
cfg.TRAIN.NOISE_STD                         = .1
# cfg.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
cfg.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
cfg.TRAIN.EPOCH_START_USE_REFINER           = 0
cfg.TRAIN.EPOCH_START_USE_MERGER            = 0
cfg.TRAIN.ENCODER_LEARNING_RATE             = 1e-3
cfg.TRAIN.DECODER_LEARNING_RATE             = 1e-3
cfg.TRAIN.REFINER_LEARNING_RATE             = 1e-3
cfg.TRAIN.MERGER_LEARNING_RATE              = 1e-4
cfg.TRAIN.ENCODER_LR_MILESTONES             = [100,200,300,400,500]
cfg.TRAIN.DECODER_LR_MILESTONES             = [100,200,300,400,500]
cfg.TRAIN.REFINER_LR_MILESTONES             = [100,200,300,400,500]
cfg.TRAIN.MERGER_LR_MILESTONES              = [100,200,300,400,500]
cfg.TRAIN.BETAS                             = (.9, .999)
cfg.TRAIN.MOMENTUM                          = .9
cfg.TRAIN.GAMMA                             = .05
cfg.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
cfg.TRAIN.UPDATE_N_VIEWS_RENDERING          = False

#
# Testing options
#
cfg.TEST                                    = edict()
# cfg.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
cfg.TEST.VOXEL_THRESH                       = [.1, .2, .3, .4, .5, .6, .7]
cfg.test_images_per_category = 10
cfg.TEST.VOXEL_THRESH_IMAGE                       = 0.2

########## Paths #################
if platform == "darwin": #local mac
    cfg.root_path = '/Users/apple/OVGU/Thesis/code/3dReconstruction'
    cfg.dataset_path ='/Users/apple/OVGU/Thesis/s2r3dfree_v1' if cfg.dataset_type == DatasetType.PIX3DSYNTHETIC1 else '/Users/apple/OVGU/Thesis/Dataset/pix3d'
    cfg.output_path = "/Users/apple/OVGU/Thesis/code/3dReconstruction/outputs/"
    cfg.home_path = "/Users/apple/Desktop/"
    cfg.empty_data_path = "/Users/apple/OVGU/Thesis/empty_room_img/"
    cfg.real_data_path = "/Users/apple/OVGU/Thesis/Dataset/pix3d/"
    cfg.num_workers = 0
    cfg.batch_size = 1
    cfg.apex = False
else:
    cfg.root_path ='/nfs1/kprabhu/3dReconstruction1/'
    cfg.dataset_path ='/nfs1/kprabhu/s2r3dfree_v1_2/' if cfg.dataset_type == DatasetType.PIX3DSYNTHETIC1 else '/nfs1/kprabhu/pix3d/'
    cfg.output_path = "/nfs1/kprabhu/outputs/"
    cfg.home_path = "/nfs1/kprabhu/"
    cfg.checkpoint_path = "/nfs1/kprabhu/outputs/" + cfg.main_name +"/"
    cfg.pretrained_checkpoint_path = "/nfs1/kprabhu/outputs/" + cfg.pretrained_name +"/"
    cfg.empty_data_path = "/nfs1/kprabhu/empty_room_img/"
    cfg.real_data_path = '/nfs1/kprabhu/pix3d/'
    cfg.num_workers = 8
    cfg.batch_size = 80


cfg.tensorboard_train = cfg.output_path + cfg.main_name  + '/tensorboard/tensorboard_training/'
cfg.tensorboard_validation = cfg.output_path + cfg.main_name  + '/tensorboard/tensorboard_validation/'
cfg.checkpoint_path = cfg.output_path + cfg.main_name + "/"
cfg.pretrained_checkpoint_path = cfg.output_path + cfg.pretrained_name + "/"
cfg.pix3d                                   = edict()
cfg.pix3d.train_indices = cfg.root_path + '/Datasets/pix3d/splits/pix3d_train_2.npy'
cfg.pix3d.test_indices = cfg.root_path + '/Datasets/pix3d/splits/pix3d_test_2.npy'
cfg.upsample = False

cfg.s2r3dfree                                   = edict()
cfg.s2r3dfree.train_test_split = cfg.root_path+"/Datasets/pix3dsynthetic/splits/train_test_split_modelwise_v1_2.p"
