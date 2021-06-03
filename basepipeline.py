"""

    Created on 10/05/21 8:36 AM 
    @author: Kartik Prabhu

"""
import os

import torch
from pytorch3d.ops import cubify
import numpy as np
from skimage.filters import threshold_otsu
from torch.utils.tensorboard import SummaryWriter

import Baseconfig
import ModelManager
from Logging import Logger
from Losses import Dice, BCE, FocalTverskyLoss, IOU
from ModelManager import LossTypes
from utils import create_binary, save_obj


class BasePipeline:
    def __init__(self,model,optimizer, config,datasetManager):
        self.config = config
        self.setup_logger()
        self.setup_dataloaders(datasetManager)

        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = config.checkpoint_path
        self.train_loss = self.get_loss(config.train_loss_type)
        self.train_loss_is_bce = config.train_loss_type == ModelManager.LossTypes.BCE
        self.num_epochs = config.num_epochs
        self.with_apex = config.apex

    def setup_dataloaders(self,datasetManager):
        dataset = datasetManager.get_dataset(self.config.dataset_type)
        traindataset = dataset.get_trainset()
        self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.config.batch_size, shuffle=True,
                                               num_workers=self.config.num_workers)
        testdataset = dataset.get_testset()
        self.test_loader = torch.utils.data.DataLoader(testdataset, batch_size=self.config.batch_size, shuffle=False,
                                              num_workers=self.config.num_workers)

    def setup_logger(self):
        self.logger = Logger(self.config.main_name, self.config.output_path + self.config.main_name).get_logger()
        self.logger.info("configuration: " + str(self.config))

        self.writer_training = SummaryWriter(self.config.tensorboard_train)
        self.writer_validating = SummaryWriter(self.config.tensorboard_validation)

    def get_loss(self,loss_type):
        switcher = {
            LossTypes.DICE: Dice(),
            LossTypes.BCE: BCE(),
            LossTypes.FOCAL_TVERSKY: FocalTverskyLoss(),
            LossTypes.IOU: IOU()
        }
        return switcher.get(loss_type, BCE())

    def save_intermediate_obj(self,istrain,training_batch_index,local_labels,output):
        process = "train" if istrain else "val"
        with torch.no_grad():


            output_path = self.checkpoint_path + self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_output.npy"
            #Pytorch error to cubify output from model, need to save it in np and then convert
            np.save(output_path, output[0].cpu().data.numpy())
            np.save(self.checkpoint_path + self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_original.npy", local_labels[0].cpu().data.numpy())

            mesh_np = np.load(output_path)
            thresh = threshold_otsu(mesh_np)
            print(thresh)
            mesh_np = (mesh_np > thresh).astype(int)
            mesh_mat = torch.tensor(mesh_np)[None,]
            output_mesh = cubify(mesh_mat, thresh=0.5)
            label_mesh = cubify(local_labels[0][None,:],thresh=0.5)

            save_obj(self.checkpoint_path +self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_output.obj",verts=output_mesh.verts_list()[0], faces=output_mesh.faces_list()[0])
            save_obj(self.checkpoint_path +self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_original.obj",verts=label_mesh.verts_list()[0], faces=label_mesh.faces_list()[0])

    def write_summary(self,writer, index, input_image, original, reconstructed, bceloss, diceLoss, diceScore, iou, writeMesh = False):
        """
        Method to write summary to the tensorboard.
        index: global_index for the visualisation
        original,reconstructer: cubified voxels [channel, Height, Width]
        Losses: all losses used as metric
        """
        print('Writing Summary...')
        writer.add_scalar('BCELoss', bceloss, index)
        writer.add_scalar('DiceLoss', diceLoss, index)
        writer.add_scalar('DiceScore', diceScore, index)
        writer.add_scalar('IOU', iou, index)
        writer.add_image('input_image', input_image, index)

        if writeMesh:
            writer.add_mesh('label', vertices=original.verts_list()[0][None,], faces=original.faces_list()[0][None,], global_step=index)
            writer.add_mesh('reconstructed', vertices=reconstructed.verts_list()[0][None,], faces=reconstructed.faces_list()[0][None,], global_step=index)

        # writer.add_image('diff', np.moveaxis(create_diff_mask(reconstructed,original,logger), -1, 0), index) #create_diff_mask is of the format HXWXC, but CXHXW is needed

    def save_model(self,CHECKPOINT_PATH, state, best_metric = False,filename='checkpoint'):
        """
        Method to save model
        """
        print('Saving model...')
        if not os.path.exists(CHECKPOINT_PATH):
            os.mkdir(CHECKPOINT_PATH)
            if best_metric:
                if not os.path.exists(CHECKPOINT_PATH + 'best_metric/'):
                    CHECKPOINT_PATH = CHECKPOINT_PATH + 'best_metric/'
                    os.mkdir(CHECKPOINT_PATH)
                    torch.save(state, CHECKPOINT_PATH + filename + str(state['epoch']) + '.pth')
