"""

    Created on 30/04/21 9:42 PM
    @author: Kartik Prabhu

"""
from easydict import EasyDict as edict
from sys import platform

cfg = edict()
config = cfg

cfg.main_name = 'test_01'
cfg.platform = platform

if platform == "darwin":
    cfg.root_path = '/Users/apple/OVGU/Thesis/code/3dReconstruction'
    cfg.dataset_path ='/Users/apple/OVGU/Thesis/SynthDataset1'
    cfg.output_path = "/Users/apple/OVGU/Thesis/code/3dReconstruction/outputs/"
    cfg.home_path = "/Users/apple/Desktop/"
    cfg.num_workers = 0
    cfg.batch_size = 4
else:
    cfg.root_path ='/nfs1/kprabhu/3dReconstruction/'
    cfg.dataset_path ='/nfs1/kprabhu/SynthDataset1/'
    cfg.output_path = "/nfs1/kprabhu/outputs/"
    cfg.home_path = "/nfs1/kprabhu/"
    cfg.checkpoint_path = "/nfs1/kprabhu/outputs/" + cfg.main_name +"/"
    cfg.num_workers = 8
    cfg.batch_size = 40


cfg.tensorboard_train = cfg.output_path + cfg.main_name  + '/tensorboard/tensorboard_training/'
cfg.tensorboard_validation = cfg.output_path + cfg.main_name  + '/tensorboard/tensorboard_validation/'
cfg.checkpoint_path = cfg.output_path + cfg.main_name + "/"

cfg.num_epochs = 1000
cfg.learning_rate = 0.1

cfg.is_mesh = False
cfg.include_depthmaps = False
cfg.include_masks = False

cfg.in_channels = 3
if cfg.include_depthmaps:
    cfg.in_channels += 1
if cfg.include_masks:
    cfg.in_channels += 1

cfg.resize = True
cfg.size = [512,512]

