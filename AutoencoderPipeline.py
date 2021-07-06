"""

    Created on 03/06/21 9:27 AM
    @author: Kartik Prabhu

"""

import torch
import torch.utils.data
from pytorch3d.ops import cubify
from torchvision.transforms import transforms

import Baseconfig
import numpy as np

import ModelManager
import transform_utils
from DatasetManager import DatasetManager
from Datasets.pix3d.Pix3dDataset import Pix3dDataset
from basepipeline import BasePipeline

# Added for Apex
from models.autoencoder import Autoencoder
from visualisation.tsneHelper import computeTSNEProjectionOfLatentSpace

try:
    import apex
    from apex import amp
except ImportError:
    print("No apex")

SEED = 2021
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

LOWEST_LOSS = 1

class AutoencoderPipeline(BasePipeline):
    def __init__(self,model,optimizer, config,datasetManager):
        super(AutoencoderPipeline, self).__init__(model, optimizer, config, datasetManager)

        # Losses
        self.dice = self.get_loss(ModelManager.LossTypes.DICE)
        self.focalTverskyLoss = self.get_loss(ModelManager.LossTypes.FOCAL_TVERSKY)
        self.iou = self.get_loss(ModelManager.LossTypes.IOU)
        self.bce = self.get_loss(ModelManager.LossTypes.BCE)


    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_training_loss = 0
            batch_index = 0
            for batch_index, (local_batch, local_labels) in enumerate(self.train_loader):

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                print('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                if torch.cuda.is_available():
                    local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
                # print(local_batch.shape)
                # print(local_labels.shape)

                # Clear gradients
                self.optimizer.zero_grad()

                dec_generated_volumes, ref_generated_volumes = self.model(local_batch)
                encoder_loss = self.train_loss(dec_generated_volumes, local_labels) * 10
                refiner_loss = self.train_loss(ref_generated_volumes, local_labels)* 10 if self.config.pix2vox_refiner else encoder_loss

                # dec_generated_volumes = torch.sigmoid(dec_generated_volumes)

                training_loss = refiner_loss
                # bceloss = self.bce(dec_generated_volumes, local_labels)
                bceloss = 0
                diceloss = self.dice(ref_generated_volumes, local_labels) if self.config.pix2vox_refiner else self.dice(dec_generated_volumes, local_labels)
                dicescore = self.dice.get_dice_score()
                iou = self.iou(ref_generated_volumes, local_labels) if self.config.pix2vox_refiner else self.iou(dec_generated_volumes, local_labels)

                # print("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + " Training..." +
                #       "\n Training_loss("+self.config.train_loss_type+"):" + str(training_loss))

                if training_batch_index % 25 == 0:  # Save best metric evaluation weights
                    self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + " Training..." +
                                     "\n Training_loss("+self.config.train_loss_type+"):" + str(training_loss))
                    self.write_summary(self.writer_training, index=training_batch_index, input_image=local_batch[0][0],
                                       input_voxel= local_labels[0], output_voxel=dec_generated_volumes[0],
                                       bceloss=bceloss,diceLoss=diceloss,diceScore=dicescore,iou=iou, writeMesh=False)

                if self.config.platform == "darwin": #debug
                    if training_batch_index % 20 == 0:
                        self.save_intermediate_obj(training_batch_index=training_batch_index, local_labels=local_labels,
                                                   outputs=ref_generated_volumes if self.config.pix2vox_refiner else dec_generated_volumes, process = "train")

                # Calculating gradients
                if self.with_apex and torch.cuda.is_available():
                    if self.config.pix2vox_refiner:
                        with amp.scale_loss(training_loss, self.optimizer) as scaled_loss:
                            with amp.scale_loss(encoder_loss, self.optimizer) as scaled_encoder_loss:
                                scaled_loss.backward()
                                scaled_encoder_loss.backward(retain_graph=True)
                    else:
                        with amp.scale_loss(training_loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()


                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1)
                else:
                    if self.config.pix2vox_refiner:
                        encoder_loss.backward(retain_graph=True)
                        training_loss.backward()
                    else:
                        encoder_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.optimizer.step()

                training_batch_index += 1
                # Initialising the average loss metrics
                total_training_loss += training_loss.detach()

            # Calculate the average loss per batch in one epoch
            total_training_loss /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n total_training_loss:" + str(total_training_loss))
            print("Epoch:" + str(epoch) + " Average Training..." +
                  "\n total_training_loss:" + str(total_training_loss))

            if torch.cuda.is_available():
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
        data_loader = self.validation_loader
        writer = self.writer_validating

        with torch.no_grad():
            for index, (batch, labels) in enumerate(data_loader):
                no_items += 1
                # Transfer to GPU
                if torch.cuda.is_available():#no nvidia on mac!!!
                    batch, labels = batch.cuda(), labels.cuda()

                # print(batch.shape)
                # print(labels.shape)

                dec_generated_volumes, ref_generated_volumes = self.model(batch)
                # outputs = torch.sigmoid(outputs)
                # print(outputs.shape)
                # bceloss += self.bce(outputs, labels).detach().item()
                # floss += self.focalTverskyLoss(outputs, labels).detach().item()

                encoder_loss = self.train_loss(dec_generated_volumes, labels) * 10
                refiner_loss = self.train_loss(ref_generated_volumes, labels)* 10 if self.config.pix2vox_refiner else encoder_loss
                training_loss = refiner_loss
                # bceloss = self.bce(dec_generated_volumes, local_labels)
                bceloss = 0
                outputs = ref_generated_volumes if self.config.pix2vox_refiner else dec_generated_volumes

                dl = self.dice(outputs, labels)
                ds = self.dice.get_dice_score()
                jaccardIndex += self.iou(outputs, labels)
                dloss += dl.detach()
                dscore += ds.detach()

        #Average the losses
        bceloss = bceloss/no_items
        dloss = dloss /no_items
        dscore = dscore / no_items
        jaccardIndex = jaccardIndex / no_items
        process = ' Validation' if validation else ' Testing'
        self.logger.info("Epoch:" + str(epoch) + process + "..." +
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

        saveMesh = (epoch % self.config.save_mesh == 0)
        with torch.no_grad():
            self.write_summary(writer, index=epoch, input_image=batch[0][0],
                               input_voxel= labels[0], output_voxel=outputs[0],
                               bceloss=bceloss,diceLoss=dloss,diceScore=dscore,iou=jaccardIndex,writeMesh=saveMesh)

        if saveMesh or epoch == self.num_epochs-1:
            self.save_intermediate_obj(training_batch_index=epoch, local_labels=labels, outputs=outputs)

        if validation:#save only for validation
            self.save_model(self.checkpoint_path, {
                'epoch': 'last',  # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved below)
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            })

            global LOWEST_LOSS
            if LOWEST_LOSS > dloss:  # Save best metric evaluation weights
                LOWEST_LOSS = dloss
                self.logger.info('Best metric... @ epoch:' + str(epoch) + ' Current Lowest loss:' + str(LOWEST_LOSS))

                self.save_model(self.checkpoint_path, {
                    'epoch': 'best',
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, best_metric=True)

        del batch, labels,dloss,dscore,jaccardIndex,bceloss,floss  # garbage collection, else training wont have any memory left



if __name__ == '__main__':
    config = Baseconfig.config

    train_transforms = transforms.Compose([
        transform_utils.Normalize(mean=config.DATASET.MEAN, std=config.DATASET.STD),
        transform_utils.ToTensor(),
    ])
    traindataset = Pix3dDataset(config).get_img_trainset(train_transforms)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=64, shuffle=True,
                                               num_workers=0)

    print(train_loader.__len__())

    testdataset = Pix3dDataset(config).get_img_testset(train_transforms)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=100, shuffle=True,
                                              num_workers=0)
    model = Autoencoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    train_loss = []
    Epochs = 1
    for epoch in range(Epochs):
        running_loss = 0.0
        for batch_index, (input_batch, input_label) in enumerate(train_loader):

            print(batch_index)
            img = input_batch
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs, encoded = model(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch+1, Epochs, loss))

    input_batch, input_label = next(iter(test_loader))
    img = input_batch
    img = img.view(img.size(0), -1)
    test_images = img
    computeTSNEProjectionOfLatentSpace(test_images, model.encoder)
