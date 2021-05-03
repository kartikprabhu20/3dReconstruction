"""

    Created on 30/04/21 9:42 PM
    @author: Kartik Prabhu

"""
from easydict import EasyDict as edict

cfg = edict()
config = cfg

cfg.root_path ='/Users/apple/OVGU/Thesis/SynthDataset1'

cfg.batch_size = 4
cfg.is_mesh = False
cfg.include_depthmaps = True
cfg.include_masks = True

cfg.resize = True
cfg.size = [224,224]
