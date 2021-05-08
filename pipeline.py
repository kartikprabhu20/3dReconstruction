"""

    Created on 01/05/21 6:12 PM
    @author: Kartik Prabhu

"""
import os
import sys
import torch
import torch.utils.data
from pytorch3d.ops import cubify

import config
import numpy as np
import torchvision.transforms as transforms

from Logging import Logger
from Losses import Dice, IOU, FocalTverskyLoss, BCE
from models.pix2voxel import pix2vox
from models.unet3d import U_Net
from pix3dsynthetic.DataExploration import get_train_test_split
from pix3dsynthetic.SyntheticPix3dDataset import SyntheticPix3d
from torch.utils.tensorboard import SummaryWriter

from utils import write_summary, save_model, save_obj

torch.manual_seed(2020)
np.random.seed(2020)
LOWEST_LOSS = 0

class Pipeline:
    def __init__(self,model,optimizer, config, logger, train_loader,test_loader,writer_training, writer_validating):
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.checkpoint_path = config.checkpoint_path

    # Losses
        self.dice = Dice()
        self.focalTverskyLoss = FocalTverskyLoss()
        self.iou = IOU()
        self.bce = BCE()

        self.num_epochs = config.num_epochs


    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_floss, total_dloss, total_jaccard_index, total_dscore, total_bceloss = 0, 0, 0, 0, 0
            batch_index = 0
            for batch_index, (local_batch, local_labels) in enumerate(train_loader):

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                print('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                if config.platform != "darwin":
                    local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

                # print(local_batch.shape)
                # print(local_labels.shape)
                # Clear gradients
                self.optimizer.zero_grad()

                # floss = torch.empty()
                # bceloss = 0
                # diceloss = torch.empty()
                # output = 0

                # try:
                    # --------------------------------------------------------------------------------------------------
                # for output in self.model(local_batch):

                output = self.model(local_batch)
                print(output)
                print(output.shape)

                # output = torch.sigmoid(self.model(local_batch))
                # print(output)
                # print(output.shape)
                # bceloss += self.bce(output, local_labels)
                diceloss,dicescore = self.dice(output, local_labels)
                print("diceloss:"+str(diceloss))
                # except Exception as error:
                #     self.logger.exception(error)
                #     print("error")
                #     print(error)
                #     sys.exit()

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + "Training..." +
                             "\n diceloss:" + str(diceloss),)
                print("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + "Training..." +
                      "\n diceloss:" + str(diceloss))
                # Calculating gradients
                diceloss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                if training_batch_index % 50 == 0:  # Save best metric evaluation weights
                    write_summary(self.writer_training, self.logger, training_batch_index, cubify(local_labels, thresh=0.5),
                              cubify(torch.sigmoid(output),thresh=0.5),
                              # 6 because in some cases we have padded with 5 which returns background
                                  diceloss, diceloss, 0, 0)


                if config.platform == "darwin":
                    if training_batch_index % 20 == 0:
                        output_mesh = cubify(torch.sigmoid(output)[0][None,:],thresh=0.5)
                        label_mesh = cubify(local_labels[0][None,:],thresh=0.5)
                        save_obj(self.checkpoint_path+ str(epoch)+"_output.obj",verts=output_mesh.verts_list()[0], faces=output_mesh.faces_list()[0])
                        save_obj(self.checkpoint_path+ str(epoch)+"_original.obj",verts=label_mesh.verts_list()[0], faces=label_mesh.faces_list()[0])
                        np.save(self.checkpoint_path+ str(epoch)+".npy", output.detach().numpy())

                training_batch_index += 1

                # Initialising the average loss metrics
                # total_floss += floss.detach().item()
                # total_bceloss += bceloss.detach().item()
                total_dloss += diceloss.detach().item()


            # Calculate the average loss per batch in one epoch
            total_dloss /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                         "\n total_dloss:" + str(total_dloss))

            if config.platform != "darwin":
                torch.cuda.empty_cache()  # to avoid memory errors
            # self.validate(training_batch_index, self.model)

            self.validate_or_test(training_batch_index)

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
        logger.debug('Validating...' if validation else 'Testing...')

        floss,binloss,dloss,dscore,jaccardIndex = 0,0,0,0,0
        no_patches = 0
        self.model.eval()
        data_loader = self.test_loader
        writer = self.writer_validating

        with torch.no_grad():
            for index, (batch, labels) in enumerate(data_loader):
                no_patches += 1
                # Transfer to GPU
                if config.platform != "darwin":
                    batch, labels = batch.cuda(), labels.cuda()

                outputs = self.model(batch)
                # binloss += bceLoss(outputs, labels)
                # outputs = torch.sigmoid(outputs)

                floss += self.focalTverskyLoss(outputs, labels).detach().item()
                dl, ds = self.dice(outputs, labels)
                dloss += dl.detach().item()
                dscore += ds.detach().item()
                jaccardIndex += self.iou(outputs, labels).detach().item()

        #Average the losses
        floss = floss/no_patches
        binloss = binloss/no_patches
        dloss = dloss /no_patches
        dscore = dscore / no_patches
        jaccardIndex = jaccardIndex / no_patches
        process = ' Validation' if validation else ' Testing'
        logger.info("Epoch:" + str(epoch) + process + "..." +
                "\n focalTverskyLoss:" + str(floss) +
                "\n binloss:" + str(binloss) +
                "\n DiceLoss:" + str(dloss) +
                "\n DiceScore:" + str(dscore) +
                "\n IOU:" + str(jaccardIndex))

        write_summary(writer, logger, epoch, cubify(labels,thresh=0.5), cubify(torch.sigmoid(outputs),thresh=0.5)
                  , floss, dloss, dscore, jaccardIndex)

        if epoch % 200 == 0:
            output_mesh = cubify(torch.sigmoid(outputs)[0][None,:],thresh=0.5)
            label_mesh = cubify(labels[0][None,:],thresh=0.5)
            save_obj(self.checkpoint_path+ str(epoch)+"_output.obj",verts=output_mesh.verts_list()[0], faces=output_mesh.faces_list()[0])
            save_obj(self.checkpoint_path+ str(epoch)+"_original.obj",verts=label_mesh.verts_list()[0], faces=label_mesh.faces_list()[0])
            np.save(self.checkpoint_path+ str(epoch)+".npy", outputs.detach().numpy())

        if validation:#save only for validation
            save_model(self.checkpoint_path, {
            'epoch': 'last',  # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved below)
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            })

            global LOWEST_LOSS
            if LOWEST_LOSS > dloss:  # Save best metric evaluation weights
                LOWEST_LOSS = dloss
                logger.info('Best metric... @ epoch:' + str(epoch) + ' Current Lowest loss:' + str(LOWEST_LOSS))

                save_model(self.checkpoint_path, {
                'epoch': 'best',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, best_metric=True)

        del batch, labels,dloss,dscore,jaccardIndex,binloss  # garbage collection, else training wont have any memory left

if __name__ == '__main__':

    config = config.config
    MODEL_NAME = config.main_name
    ROOT_PATH = config.root_path
    OUTPUT_PATH = config.output_path
    LOGGER_PATH = OUTPUT_PATH +MODEL_NAME


    logger = Logger(MODEL_NAME, LOGGER_PATH).get_logger()

    train_img_list,train_model_list,test_img_list,test_model_list = get_train_test_split(ROOT_PATH+"/pix3dsynthetic/train_test_split.p")

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