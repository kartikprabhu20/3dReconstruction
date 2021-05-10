"""

    Created on 01/05/21 6:12 PM
    @author: Kartik Prabhu

"""
import os
import sys
import torch
import torch.utils.data
from pytorch3d.ops import cubify

import Baseconfig
import numpy as np
import torchvision.transforms as transforms

from Logging import Logger
from Losses import Dice, IOU, FocalTverskyLoss, BCE
from basepipeline import BasePipeline
from models.pix2voxel import pix2vox
from models.unet3d import U_Net
from pix3dsynthetic.DataExploration import get_train_test_split
from pix3dsynthetic.SyntheticPix3dDataset import SyntheticPix3d
from torch.utils.tensorboard import SummaryWriter

from utils import save_obj, create_binary

torch.manual_seed(2020)
np.random.seed(2020)
LOWEST_LOSS = 1

class Pipeline(BasePipeline):
    def __init__(self,model,optimizer, config, logger, train_loader,test_loader,writer_training, writer_validating):
        super(Pipeline,self).__init__(model,optimizer, config, logger, train_loader,test_loader,writer_training, writer_validating)

        # Losses
        self.dice = self.get_loss(Baseconfig.LossTypes.DICE)
        self.focalTverskyLoss = self.get_loss(Baseconfig.LossTypes.FOCAL_TVERSKY)
        self.iou = self.get_loss(Baseconfig.LossTypes.IOU)
        self.bce = self.get_loss(Baseconfig.LossTypes.BCE)


    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_training_loss = 0
            batch_index = 0
            for batch_index, (local_batch, local_labels) in enumerate(train_loader):

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                # print('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                if self.config.platform != "darwin":
                    local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
                # print(local_batch.shape)
                # print(local_labels.shape)

                # Clear gradients
                self.optimizer.zero_grad()

                output = self.model(local_batch)
                # output = torch.sigmoid(output)
                # print(output)
                # print(output.shape)

                training_loss = self.train_loss(output, local_labels)
                bceloss = self.bce(output, local_labels)
                diceloss = self.dice(output, local_labels)
                dicescore = self.dice.get_dice_score()

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + "Training..." +
                             "\n Training_loss("+self.config.train_loss_type+"):" + str(training_loss),)
                # print("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + "Training..." +
                #       "\n diceloss:" + str(diceloss))

                # Calculating gradients
                training_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                if training_batch_index % 50 == 0:  # Save best metric evaluation weights
                    with torch.no_grad():
                        self.write_summary(self.writer_training, training_batch_index, cubify(local_labels, thresh=0.5),
                                  cubify(torch.sigmoid(output), thresh=0.5),
                                  # 6 because in some cases we have padded with 5 which returns background
                                  bceloss,diceloss, dicescore, 0)

                if self.config.platform == "darwin":
                    if training_batch_index % 20 == 0:
                        self.save_intermediate_obj(istrain=True,training_batch_index=training_batch_index,local_labels=local_labels,output=output)

                training_batch_index += 1

                # Initialising the average loss metrics
                total_training_loss += training_loss.detach().item()


            # Calculate the average loss per batch in one epoch
            total_training_loss /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                         "\n total_training_loss:" + str(total_training_loss))
            print("Epoch:" + str(epoch) + " Average Training..." +
                  "\n total_training_loss:" + str(total_training_loss))

            if self.config.platform != "darwin":
                torch.cuda.empty_cache()  # to avoid memory errors

            self.validate_or_test(epoch)

        return self.model

    def validate_or_test(self,epoch, validation = True):
        '''
        Method to test or validate
        :param epoch: Epoch after which validation is performed(can be anything for test)
        :param currentModel: model for which validation/test is being performed
        :param LOWEST_LOSS:
        :param validation: True for validation(default), False for testing
        :return:
        '''
        self.logger.debug('Validating...' if validation else 'Testing...')
        print("Validating...")

        bceloss,floss,dloss,dscore,jaccardIndex = 0,0,0,0,0
        no_items = 0
        self.model.eval()
        data_loader = self.test_loader
        writer = self.writer_validating

        with torch.no_grad():
            for index, (batch, labels) in enumerate(data_loader):
                no_items += self.config.batch_size
                # Transfer to GPU
                if self.config.platform != "darwin":
                    batch, labels = batch.cuda(), labels.cuda()

                outputs = self.model(batch)
                # binloss += bceLoss(outputs, labels)
                # outputs = torch.sigmoid(outputs)

                bceloss += self.bce(outputs, labels).detach().item()
                # floss += self.focalTverskyLoss(outputs, labels).detach().item()
                dl = self.dice(outputs, labels)
                ds = self.dice.get_dice_score()
                dloss += dl.detach().item()
                dscore += ds.detach().item()
                jaccardIndex += self.iou(outputs, labels).detach().item()

        #Average the losses
        bceloss = bceloss/no_items
        # floss = floss/no_items
        bceloss = bceloss/no_items
        dloss = dloss /no_items
        dscore = dscore / no_items
        jaccardIndex = jaccardIndex / no_items
        process = ' Validation' if validation else ' Testing'
        logger.info("Epoch:" + str(epoch) + process + "..." +
                "\n focalTverskyLoss:" + str(floss) +
                "\n binloss:" + str(bceloss) +
                "\n DiceLoss:" + str(dloss) +
                "\n DiceScore:" + str(dscore) +
                "\n IOU:" + str(jaccardIndex))

        print("Epoch:" + str(epoch) + process + "..." +
              "\n focalTverskyLoss:" + str(floss) +
              "\n binloss:" + str(bceloss) +
              "\n DiceLoss:" + str(dloss) +
              "\n DiceScore:" + str(dscore) +
              "\n IOU:" + str(jaccardIndex))

        with torch.no_grad():
            self.write_summary(writer, epoch, cubify(labels[0][None,:],thresh=0.5), cubify(torch.sigmoid(outputs)[0][None,:],thresh=0.5)
                          , bceloss, dloss, dscore, jaccardIndex)

        if epoch % 20 == 0 or epoch == self.num_epochs-1:
            self.save_intermediate_obj(istrain=False,training_batch_index=epoch,local_labels=labels,output=outputs)

        if validation:#save only for validation
            self.save_model(self.checkpoint_path, {
            'epoch': 'last',  # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved below)
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            })

            global LOWEST_LOSS
            if LOWEST_LOSS > dloss:  # Save best metric evaluation weights
                LOWEST_LOSS = dloss
                logger.info('Best metric... @ epoch:' + str(epoch) + ' Current Lowest loss:' + str(LOWEST_LOSS))

                self.save_model(self.checkpoint_path, {
                'epoch': 'best',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, best_metric=True)

        del batch, labels,dloss,dscore,jaccardIndex,bceloss,floss  # garbage collection, else training wont have any memory left


if __name__ == '__main__':

    config = Baseconfig.config
    print("configuration: " + str(config))

    MODEL_NAME = config.main_name
    ROOT_PATH = config.root_path
    OUTPUT_PATH = config.output_path
    LOGGER_PATH = OUTPUT_PATH +MODEL_NAME


    logger = Logger(MODEL_NAME, LOGGER_PATH).get_logger()
    logger.info("configuration: " + str(config))

    train_img_list,train_model_list,test_img_list,test_model_list = get_train_test_split(ROOT_PATH+"/pix3dsynthetic/train_test_split.p")
    # print(train_img_list)
    # print(test_img_list)

    writer_training = SummaryWriter(config.tensorboard_train)
    writer_validating = SummaryWriter(config.tensorboard_validation)

    # model = U_Net()
    model = pix2vox(in_channel=config.in_channels)
    if config.platform != "darwin":
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    traindataset = SyntheticPix3d(config,train_img_list,train_model_list)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.num_workers)

    testdataset = SyntheticPix3d(config,test_img_list,test_model_list)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=config.batch_size, shuffle=True,
                                           num_workers=config.num_workers)

    pipeline = Pipeline(model=model,optimizer=optimizer,config=config,train_loader= train_loader,test_loader=test_loader,logger=logger, writer_training= writer_training, writer_validating=writer_validating)
    pipeline.train()
