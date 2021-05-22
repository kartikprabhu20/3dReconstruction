"""

    Created on 10/05/21 8:36 AM 
    @author: Kartik Prabhu

"""
import os

import torch
from pytorch3d.ops import cubify
import numpy as np

import Baseconfig
from Losses import Dice, BCE, FocalTverskyLoss, IOU
from Baseconfig import LossTypes
from utils import create_binary, save_obj


class BasePipeline:
    def __init__(self,model,optimizer, config, logger, train_loader,test_loader,writer_training, writer_validating):
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.checkpoint_path = config.checkpoint_path
        self.config = config
        self.train_loss = self.get_loss(config.train_loss_type)
        self.train_loss_is_bce = config.train_loss_type == Baseconfig.LossType.BCE
        self.num_epochs = config.num_epochs
        self.with_apex = config.apex

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

            output_mesh = cubify(torch.sigmoid(output)[0][None,:],thresh=0.5)
            label_mesh = cubify(local_labels[0][None,:],thresh=0.5)

            # save_obj(self.checkpoint_path +self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_output.obj",verts=output_mesh.verts_list()[0], faces=output_mesh.faces_list()[0])
            # save_obj(self.checkpoint_path +self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_original.obj",verts=label_mesh.verts_list()[0], faces=label_mesh.faces_list()[0])
            np.save(self.checkpoint_path + self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_output.npy", output[0].cpu().data.numpy())
            np.save(self.checkpoint_path + self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_original.npy", local_labels[0].cpu().data.numpy())

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
