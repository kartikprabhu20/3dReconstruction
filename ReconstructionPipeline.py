"""

    Created on 01/05/21 6:12 PM
    @author: Kartik Prabhu

"""
import os

import torch
import torch.utils.data
from pytorch3d.ops import cubify

import Baseconfig
import numpy as np

import ModelManager
import network_utils
import utils
from DatasetManager import DatasetManager, DatasetType
from basepipeline import BasePipeline

# Added for Apex
from models.pix2voxel import Encoder, Decoder, Refiner, Merger
from datetime import datetime as dt
from torchvision.utils import save_image

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
best_iou = -1
best_epoch = -1

class ReconstructionPipeline(BasePipeline):
    def __init__(self, config,datasetManager):
        super(ReconstructionPipeline, self).__init__(config, datasetManager)

        # Losses
        self.dice = self.get_loss(ModelManager.LossTypes.DICE)
        self.focalTverskyLoss = self.get_loss(ModelManager.LossTypes.FOCAL_TVERSKY)
        self.iou = self.get_loss(ModelManager.LossTypes.IOU)
        self.bce = self.get_loss(ModelManager.LossTypes.BCE)

        self.pix2vox,_ = ModelManager.ModelManager(self.config).get_Model(self.config.model_type)
        self.init_models(load_model=config.load_model)

    def init_models(self,load_model):
        # Set up networks
        self.encoder = self.pix2vox.encoder
        self.decoder = self.pix2vox.decoder
        self.encoder.apply(network_utils.init_weights)
        self.decoder.apply(network_utils.init_weights)

        print('Parameters in Encoder: %d.' % (utils.count_parameters(self.encoder)))
        print('Parameters in Decoder: %d.' % (utils.count_parameters(self.decoder)))

        # if config.pix2vox_refiner:
        self.refiner = self.pix2vox.refiner
        self.refiner.apply(network_utils.init_weights)
        print('Parameters in Refiner: %d.' % (utils.count_parameters(self.refiner)))

        # if config.pix2vox_merger:
        self.merger =  self.pix2vox.merger
        print('Parameters in Merger: %d.' % (utils.count_parameters(self.merger)))
        self.merger.apply(network_utils.init_weights)

        # Set up solver
        if self.config.optimizer_type == ModelManager.OptimizerType.ADAM:
            self.encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                                   lr=self.config.TRAIN.ENCODER_LEARNING_RATE,
                                                   betas=self.config.TRAIN.BETAS)
            self.decoder_solver = torch.optim.Adam(self.decoder.parameters(),
                                                   lr=self.config.TRAIN.DECODER_LEARNING_RATE,
                                                   betas=self.config.TRAIN.BETAS)
            self.refiner_solver = torch.optim.Adam(self.refiner.parameters(),
                                                   lr=self.config.TRAIN.REFINER_LEARNING_RATE,
                                                   betas=self.config.TRAIN.BETAS)
            self.merger_solver = torch.optim.Adam(self.merger.parameters(), lr=self.config.TRAIN.MERGER_LEARNING_RATE, betas=self.config.TRAIN.BETAS)
        elif self.config.optimizer_type == ModelManager.OptimizerType.SGD:
            self.encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                                  lr=self.config.TRAIN.ENCODER_LEARNING_RATE,
                                                  momentum=self.config.TRAIN.MOMENTUM)
            self.decoder_solver = torch.optim.SGD(self.decoder.parameters(),
                                                  lr=self.config.TRAIN.DECODER_LEARNING_RATE,
                                                  momentum=self.config.TRAIN.MOMENTUM)
            self.refiner_solver = torch.optim.SGD(self.refiner.parameters(),
                                                  lr=self.config.TRAIN.REFINER_LEARNING_RATE,
                                                  momentum=self.config.TRAIN.MOMENTUM)
            self.merger_solver = torch.optim.SGD(self.merger.parameters(),
                                                 lr=self.config.TRAIN.MERGER_LEARNING_RATE,
                                                 momentum=self.config.TRAIN.MOMENTUM)
        else:
            raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), self.config.TRAIN.POLICY))

        if load_model:
            self.encoder,self.encoder_solver,self.decoder,self.decoder_solver,\
            self.refiner,self.refiner_solver,self.merger,self.merger_solver = network_utils.load_checkpoint(self.config,self.encoder,self.encoder_solver,self.decoder,
                                                                                  self.decoder_solver,self.refiner,self.refiner_solver,self.merger,self.merger_solver)

    def train(self):
        self.logger.debug("Training...")



        # Set up learning rate scheduler to decay learning rates dynamically
        encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.encoder_solver,
                                                                milestones=self.config.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=self.config.TRAIN.GAMMA)
        decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.decoder_solver,
                                                                milestones=self.config.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=self.config.TRAIN.GAMMA)
        refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.refiner_solver,
                                                                milestones=self.config.TRAIN.REFINER_LR_MILESTONES,
                                                                gamma=self.config.TRAIN.GAMMA)
        merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.merger_solver,
                                                               milestones=self.config.TRAIN.MERGER_LR_MILESTONES,
                                                               gamma=self.config.TRAIN.GAMMA)

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.refiner = self.refiner.cuda()
            self.merger = self.merger.cuda()

        training_batch_index = 0

        for epoch in range(self.num_epochs):
            # self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            self.encoder.train()
            self.decoder.train()
            self.refiner.train()
            self.merger.train()
            total_training_loss = 0
            batch_index = 0
            for batch_index, (taxonomy_id, local_batch, local_labels) in enumerate(self.train_loader):

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                print('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                if torch.cuda.is_available():
                    local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
                # print(local_batch.shape)
                # print(local_labels.shape)

                # Clear gradients
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.refiner.zero_grad()
                self.merger.zero_grad()

                # dec_generated_volumes, ref_generated_volumes = self.model(local_batch)
                # encoder_loss = self.train_loss(dec_generated_volumes, local_labels) * 10
                # refiner_loss = self.train_loss(ref_generated_volumes, local_labels)* 10 if self.config.pix2vox_refiner else encoder_loss

                # dec_generated_volumes = torch.sigmoid(dec_generated_volumes)

                # Train the encoder, decoder, refiner, and merger
                image_features = self.encoder(local_batch)
                raw_features, dec_generated_volumes = self.decoder(image_features)

                if self.config.pix2vox_merger:
                    dec_generated_volumes = self.merger(raw_features, dec_generated_volumes)
                else:
                    dec_generated_volumes = torch.mean(dec_generated_volumes, dim=1)
                encoder_loss = self.bce(dec_generated_volumes, local_labels) * 10

                if self.config.pix2vox_refiner:
                    ref_generated_volumes = self.refiner(dec_generated_volumes)
                    refiner_loss = self.bce(ref_generated_volumes, local_labels) * 10
                else:
                    refiner_loss = encoder_loss


                training_loss = refiner_loss
                bceloss = refiner_loss
                # bceloss = self.bce(dec_generated_volumes, local_labels)
                # bceloss = 0
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
                        self.save_intermediate_obj(process="Train", training_batch_index=training_batch_index, input_images=local_batch,local_labels=local_labels,
                                                   outputs=ref_generated_volumes if self.config.pix2vox_refiner else dec_generated_volumes)

                if self.config.pix2vox_refiner:
                    encoder_loss.backward(retain_graph=True)
                    refiner_loss.backward()
                else:
                    encoder_loss.backward()

                self.encoder_solver.step()
                self.decoder_solver.step()
                self.refiner_solver.step()
                self.merger_solver.step()

                # Calculating gradients
                # if self.with_apex and torch.cuda.is_available():
                #     if self.config.pix2vox_refiner:
                #         with amp.scale_loss(training_loss, self.optimizer) as scaled_loss:
                #             with amp.scale_loss(encoder_loss, self.optimizer) as scaled_encoder_loss:
                #                 scaled_loss.backward()
                #                 scaled_encoder_loss.backward(retain_graph=True)
                #     else:
                #         with amp.scale_loss(training_loss, self.optimizer) as scaled_loss:
                #             scaled_loss.backward()
                #
                #
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1)
                # else:
                #     if self.config.pix2vox_refiner:
                #         encoder_loss.backward(retain_graph=True)
                #         training_loss.backward()
                #     else:
                #         encoder_loss.backward()
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                #
                # self.optimizer.step()

                training_batch_index += 1
                # Initialising the average loss metrics
                total_training_loss += training_loss.detach()

            # Adjust learning rate
            encoder_lr_scheduler.step()
            decoder_lr_scheduler.step()
            refiner_lr_scheduler.step()
            merger_lr_scheduler.step()

            # Calculate the average loss per batch in one epoch
            total_training_loss /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                         "\n total_training_loss:" + str(total_training_loss))
            print("Epoch:" + str(epoch) + " Average Training..." +
                  "\n total_training_loss:" + str(total_training_loss))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # to avoid memory errors

            self.validate_or_test(epoch,self.encoder,self.decoder,self.refiner,self.merger,self.encoder_solver,self.decoder_solver,self.refiner_solver,self.merger_solver)

            if epoch % self.config.save_mesh == 0:
                self.realdata_test()

    def validate_or_test(self,epoch,encoder=None,
                         decoder=None,
                         refiner=None,
                         merger=None,
                         encoder_solver=None,
                         decoder_solver=None,
                         refiner_solver=None,
                         merger_solver=None,
                         validation = True):
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
        encoder_losses = network_utils.AverageMeter()
        refiner_losses = network_utils.AverageMeter()
        # self.model.eval()
        data_loader = self.validation_loader
        writer = self.writer_validating
        n_samples = len(self.validation_loader)

        # Switch models to evaluation mode
        encoder.eval()
        decoder.eval()
        refiner.eval()
        merger.eval()

        test_iou_dict = dict()
        with torch.no_grad():
            for index, (taxonomy_ids, batch, labels) in enumerate(data_loader):
                # Transfer to GPU
                if torch.cuda.is_available():#no nvidia on mac!!!
                    batch, labels = batch.cuda(), labels.cuda()

                # print(batch.shape)
                # print(labels.shape)

                # dec_generated_volumes, ref_generated_volumes = self.model(batch)
                # # outputs = torch.sigmoid(outputs)
                # # print(outputs.shape)
                # # bceloss += self.bce(outputs, labels).detach().item()
                # # floss += self.focalTverskyLoss(outputs, labels).detach().item()
                #
                # encoder_loss = self.train_loss(dec_generated_volumes, labels) * 10
                # refiner_loss = self.train_loss(ref_generated_volumes, labels)* 10 if self.config.pix2vox_refiner else encoder_loss
                # training_loss = refiner_loss
                # # bceloss = 0
                # outputs = ref_generated_volumes if self.config.pix2vox_refiner else dec_generated_volumes


                image_features = encoder(batch)
                raw_features, generated_volume = decoder(image_features)

                if self.config.pix2vox_merger:
                    generated_volume = merger(raw_features, generated_volume)
                else:
                    generated_volume = torch.mean(generated_volume, dim=1)
                encoder_loss = self.bce(generated_volume, labels) * 10

                if self.config.pix2vox_refiner:
                    generated_volume = refiner(generated_volume)
                    refiner_loss = self.bce(generated_volume, labels) * 10
                else:
                    refiner_loss = encoder_loss

                outputs = generated_volume

                bceloss + self.bce(outputs, labels)
                dl = self.dice(outputs, labels)
                ds = self.dice.get_dice_score()
                jaccardIndex += self.iou(outputs, labels)
                dloss += dl.detach()
                dscore += ds.detach()

                encoder_losses.update(encoder_loss.item())
                refiner_losses.update(refiner_loss.item())

                for i in range(0, batch.shape[0]):
                    sample_iou = []
                    for th in self.config.TEST.VOXEL_THRESH:
                        _volume = torch.ge(outputs[i], th).float()
                        sample_iou.append(self.iou(_volume, labels[i]).item())

                    if taxonomy_ids[i] not in test_iou_dict:
                        test_iou_dict[taxonomy_ids[i]] = {'n_samples': 0, 'iou': [0] * len(self.config.TEST.VOXEL_THRESH)}
                    test_iou_dict[taxonomy_ids[i]]['n_samples'] += 1
                    test_iou_dict[taxonomy_ids[i]]['iou']  = [a + b for a, b in zip(test_iou_dict[taxonomy_ids[i]]['iou'], sample_iou)]

        #Average the losses
        bceloss = bceloss/n_samples
        dloss = dloss /n_samples
        dscore = dscore / n_samples
        jaccardIndex = jaccardIndex / n_samples
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
            self.save_intermediate_obj(training_batch_index=epoch,input_images=batch, local_labels=labels, outputs=outputs)

        global LOWEST_LOSS
        global best_iou
        global best_epoch

        if validation:#save only for validation
            # self.save_model(self.checkpoint_path, {
            # 'epoch': 'last',  # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved below)
            # 'state_dict': self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict()
            # })
            network_utils.save_checkpoints(self.config, self.checkpoint_path,
                                                 'last', encoder, encoder_solver, decoder, decoder_solver,
                                                 refiner, refiner_solver, merger, merger_solver, best_iou, best_epoch)



        if best_iou < jaccardIndex:  # Save best metric evaluation weights
                best_iou = jaccardIndex
                best_epoch = epoch
                self.logger.info('Best metric... @ epoch:' + str(epoch) + ' Current Lowest loss:' + str(LOWEST_LOSS))
                self.logger.info('Best metric... @ epoch:' + str(epoch) + ' Current best_iou:' + str(best_iou))


                network_utils.save_checkpoints(self.config, self.checkpoint_path,
                                               'best', encoder, encoder_solver, decoder, decoder_solver,
                                               refiner, refiner_solver, merger, merger_solver, best_iou, best_epoch)

                # self.save_model(self.checkpoint_path, {
                # 'epoch': 'best',
                # 'state_dict': self.model.state_dict(),
                # 'optimizer': self.optimizer.state_dict()}, best_metric=True)

        del batch, labels,dloss,dscore,jaccardIndex,bceloss,floss  # garbage collection, else training wont have any memory left

        # Output testing results
        for taxonomy_id in test_iou_dict:
            test_iou_dict[taxonomy_id]['iou']  = np.divide(test_iou_dict[taxonomy_id]['iou'], test_iou_dict[taxonomy_id]['n_samples'])
            # print(test_iou_dict[taxonomy_id]['iou'])
            # print("\n")

        # Print header
        self.logger.info('============================ Validation RESULTS ============================')
        self.logger.info('Taxonomy\t #Sample\t\n')
        for th in self.config.TEST.VOXEL_THRESH:
            self.logger.info('t=%.2f \t' % th)
        self.logger.info(" ======================================================================")
        # Print body
        for taxonomy_id in test_iou_dict:
            self.logger.info(taxonomy_id+":\t")
            self.logger.info('%d \t' % test_iou_dict[taxonomy_id]['n_samples'])
            for ti in test_iou_dict[taxonomy_id]['iou']:
                self.logger.info('%.4f \t' % ti)
            self.logger.info(" ======================================================================")

        # Print mean IoU for each threshold
        mean_iou = [0] * len(self.config.TEST.VOXEL_THRESH)
        self.logger.info('Overall:\t')
        self.logger.info('%d \t' % n_samples)
        for i in range(0,len(self.config.TEST.VOXEL_THRESH)):
            for taxonomy_id in test_iou_dict:
                mean_iou[i] += test_iou_dict[taxonomy_id]['iou'][i]

        mean_iou  = np.divide(mean_iou, n_samples)
        for mi in mean_iou:
            self.logger.info('%.4f \t' % mi)

        # print('============================ TEST RESULTS ============================')
        # print('Taxonomy\t #Sample\t', end='')
        # for th in self.config.TEST.VOXEL_THRESH:
        #     print('t=%.2f' % th, end='\t')
        # print()
        # # Print body
        # for taxonomy_id in test_iou_dict:
        #     print(taxonomy_id+":", end='\t')
        #     print('%d' % test_iou_dict[taxonomy_id]['n_samples'], end='\t')
        #     for ti in test_iou_dict[taxonomy_id]['iou']:
        #         print('%.4f' % ti, end='\t')
        #     print()
        #
        # # Print mean IoU for each threshold
        # mean_iou = [0] * len(self.config.TEST.VOXEL_THRESH)
        # print('Overall:', end='\t')
        # print('%d' % n_samples, end='\t')
        # for i in range(0,len(self.config.TEST.VOXEL_THRESH)):
        #     for taxonomy_id in test_iou_dict:
        #         mean_iou[i] += test_iou_dict[taxonomy_id]['iou'][i]
        #
        # mean_iou  = np.divide(mean_iou, n_samples)
        # for mi in mean_iou:
        #     print('%.4f' % mi, end='\t')

    def validate(self):
        self.validate_or_test(0,self.encoder,self.decoder,self.refiner,self.merger,self.encoder_solver,self.decoder_solver,self.refiner_solver,self.merger_solver)


    def test(self, empty_test = False):
        dataset = self.datasetManager.get_dataset(self.config.dataset_type)
        testdataset = dataset.get_testset(transforms=self.test_transforms,images_per_category=self.config.test_images_per_category)
        data_loader = torch.utils.data.DataLoader(testdataset, batch_size=self.config.batch_size, shuffle=False,
                                               num_workers=self.config.num_workers, pin_memory=True)

        n_samples = len(data_loader)
        # Switch models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        self.refiner.eval()
        self.merger.eval()

        test_iou_dict = dict()
        with torch.no_grad():
            for index, (taxonomy_ids, batch, labels) in enumerate(data_loader):
                # Transfer to GPU
                if torch.cuda.is_available():#no nvidia on mac!!!
                    batch, labels = batch.cuda(), labels.cuda()

                image_features = self.encoder(batch)
                raw_features, generated_volume = self.decoder(image_features)

                if self.config.pix2vox_merger:
                    generated_volume = self.merger(raw_features, generated_volume)
                else:
                    generated_volume = torch.mean(generated_volume, dim=1)

                if self.config.pix2vox_refiner:
                    generated_volume = self.refiner(generated_volume)

                outputs = generated_volume

                self.save_intermediate_obj(training_batch_index=index,input_images=batch, local_labels=labels, outputs=outputs, process = "empty" if empty_test else "test", threshold=self.config.TEST.VOXEL_THRESH_IMAGE)

                for i in range(0, batch.shape[0]):
                    sample_iou = []
                    for th in self.config.TEST.VOXEL_THRESH:
                        _volume = torch.ge(outputs[i], th).float()
                        sample_iou.append(self.iou(_volume, labels[i]).item())

                    if taxonomy_ids[i] not in test_iou_dict:
                        test_iou_dict[taxonomy_ids[i]] = {'n_samples': 0, 'iou': [0] * len(self.config.TEST.VOXEL_THRESH)}
                    test_iou_dict[taxonomy_ids[i]]['n_samples'] += 1
                    test_iou_dict[taxonomy_ids[i]]['iou']  = [a + b for a, b in zip(test_iou_dict[taxonomy_ids[i]]['iou'], sample_iou)]

        # Output testing results
        for taxonomy_id in test_iou_dict:
            test_iou_dict[taxonomy_id]['iou']  = np.divide(test_iou_dict[taxonomy_id]['iou'], test_iou_dict[taxonomy_id]['n_samples'])
            # print(test_iou_dict[taxonomy_id]['iou'])
            # print("\n")

        # Print header
        self.logger.info('============================ TEST RESULTS ============================')
        self.logger.info('Taxonomy\t #Sample\t\n')
        for th in self.config.TEST.VOXEL_THRESH:
            self.logger.info('t=%.2f \t' % th)
        self.logger.info(" ======================================================================")
        # Print body
        for taxonomy_id in test_iou_dict:
            self.logger.info(taxonomy_id+":\t")
            self.logger.info('%d \t' % test_iou_dict[taxonomy_id]['n_samples'])
            for ti in test_iou_dict[taxonomy_id]['iou']:
                self.logger.info('%.4f \t' % ti)
            self.logger.info(" ======================================================================")

        # Print mean IoU for each threshold
        mean_iou = [0] * len(self.config.TEST.VOXEL_THRESH)
        self.logger.info('Overall:\t')
        self.logger.info('%d \t' % n_samples)
        for i in range(0,len(self.config.TEST.VOXEL_THRESH)):
            for taxonomy_id in test_iou_dict:
                mean_iou[i] += test_iou_dict[taxonomy_id]['iou'][i]

        mean_iou  = np.divide(mean_iou, n_samples)
        for mi in mean_iou:
            self.logger.info('%.4f \t' % mi)

    def empty_test(self):

        empty_dataset = self.datasetManager.get_dataset(DatasetType.EMPTY)
        emptydataset = empty_dataset.get_img_testset(transforms=self.test_transforms,images_per_category=self.config.test_images_per_category)
        data_loader = torch.utils.data.DataLoader(emptydataset, batch_size=self.config.batch_size, shuffle=False,
                                                        num_workers=self.config.num_workers, pin_memory=True)

        # Switch models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        self.refiner.eval()
        self.merger.eval()

        with torch.no_grad():
            for index, batch in enumerate(data_loader):
                # Transfer to GPU
                if torch.cuda.is_available():#no nvidia on mac!!!
                    batch = batch.cuda()

                image_features = self.encoder(batch)
                raw_features, generated_volume = self.decoder(image_features)

                if self.config.pix2vox_merger:
                    generated_volume = self.merger(raw_features, generated_volume)
                else:
                    generated_volume = torch.mean(generated_volume, dim=1)

                if self.config.pix2vox_refiner:
                    generated_volume = self.refiner(generated_volume)

                outputs = generated_volume

                output_path = self.checkpoint_path + self.config.main_name+"_"+ str(index)+"_"+"empty"+"_output.npy"

                output_np = outputs[0].detach().cpu().numpy()
                output_np = output_np > 0.2
                output = torch.tensor(output_np)
                self.gen_plot(output, savefig=True, path =self.checkpoint_path + self.config.main_name + "_" + str(index) + "_" + "empty" + "_output_" + str(index) + ".png")

                # save_image(input_images[i][0],self.checkpoint_path + self.config.main_name + "_" + str(index) + "_" + "empty" + "_input_" + str(index) + ".png")

    def realdata_test(self):
        self.config.dataset_path = self.config.real_data_path
        dataset = self.datasetManager.get_dataset(DatasetType.PIX3D)
        dataset = dataset.get_testset(transforms=self.test_transforms,images_per_category=self.config.test_images_per_category)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False,
                                                  num_workers=self.config.num_workers, pin_memory=True)
        n_samples = len(data_loader)
        # Switch models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        self.refiner.eval()
        self.merger.eval()

        test_iou_dict = dict()
        with torch.no_grad():
            for index, (taxonomy_ids, batch, labels) in enumerate(data_loader):
                # Transfer to GPU
                if torch.cuda.is_available():#no nvidia on mac!!!
                    batch, labels = batch.cuda(), labels.cuda()

                image_features = self.encoder(batch)
                raw_features, generated_volume = self.decoder(image_features)

                if self.config.pix2vox_merger:
                    generated_volume = self.merger(raw_features, generated_volume)
                else:
                    generated_volume = torch.mean(generated_volume, dim=1)

                if self.config.pix2vox_refiner:
                    generated_volume = self.refiner(generated_volume)

                outputs = generated_volume

                output_path = self.checkpoint_path + self.config.main_name+"_"+ str(index)+"_"+"empty"+"_output.npy"

                output_np = outputs[0].detach().cpu().numpy()
                output_np = output_np > 0.2
                output = torch.tensor(output_np)

                for i in range(0, batch.shape[0]):
                    sample_iou = []
                    for th in self.config.TEST.VOXEL_THRESH:
                        _volume = torch.ge(outputs[i], th).float()
                        sample_iou.append(self.iou(_volume, labels[i]).item())

                    if taxonomy_ids[i] not in test_iou_dict:
                        test_iou_dict[taxonomy_ids[i]] = {'n_samples': 0, 'iou': [0] * len(self.config.TEST.VOXEL_THRESH)}
                    test_iou_dict[taxonomy_ids[i]]['n_samples'] += 1
                    test_iou_dict[taxonomy_ids[i]]['iou']  = [a + b for a, b in zip(test_iou_dict[taxonomy_ids[i]]['iou'], sample_iou)]

        # Output testing results
        for taxonomy_id in test_iou_dict:
            test_iou_dict[taxonomy_id]['iou']  = np.divide(test_iou_dict[taxonomy_id]['iou'], test_iou_dict[taxonomy_id]['n_samples'])
            # print(test_iou_dict[taxonomy_id]['iou'])
            # print("\n")

        # Print header
        self.logger.info('============================ REAL DATA RESULTS ============================')
        self.logger.info('Taxonomy\t #Sample\t\n')
        for th in self.config.TEST.VOXEL_THRESH:
            self.logger.info('t=%.2f \t' % th)
        self.logger.info(" ======================================================================")
        # Print body
        for taxonomy_id in test_iou_dict:
            self.logger.info(taxonomy_id+":\t")
            self.logger.info('%d \t' % test_iou_dict[taxonomy_id]['n_samples'])
            for ti in test_iou_dict[taxonomy_id]['iou']:
                self.logger.info('%.4f \t' % ti)
            self.logger.info(" ======================================================================")

        # Print mean IoU for each threshold
        mean_iou = [0] * len(self.config.TEST.VOXEL_THRESH)
        self.logger.info('Overall:\t')
        self.logger.info('%d \t' % n_samples)
        for i in range(0,len(self.config.TEST.VOXEL_THRESH)):
            for taxonomy_id in test_iou_dict:
                mean_iou[i] += test_iou_dict[taxonomy_id]['iou'][i]

        mean_iou  = np.divide(mean_iou, n_samples)
        for mi in mean_iou:
            self.logger.info('%.4f \t' % mi)


if __name__ == '__main__':

    config = Baseconfig.config
    print("configuration: " + str(config))

    model, optimizer = ModelManager.ModelManager(config).get_Model(config.model_type)

    pipeline = ReconstructionPipeline(config=config, datasetManager=DatasetManager(config))
    encoder = Encoder(config=config,in_channel=config.in_channels)
    decoder = Decoder(config=config)
    refiner = Refiner(config=config)
    merger =  Merger(config=config)
    encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                      lr=config.TRAIN.ENCODER_LEARNING_RATE,
                                      betas=config.TRAIN.BETAS)
    decoder_solver = torch.optim.Adam(decoder.parameters(),
                                      lr=config.TRAIN.DECODER_LEARNING_RATE,
                                      betas=config.TRAIN.BETAS)
    refiner_solver = torch.optim.Adam(refiner.parameters(),
                                      lr=config.TRAIN.REFINER_LEARNING_RATE,
                                      betas=config.TRAIN.BETAS)
    merger_solver = torch.optim.Adam(merger.parameters(), lr=config.TRAIN.MERGER_LEARNING_RATE, betas=config.TRAIN.BETAS)

    pipeline.validate_or_test(1,encoder,decoder,refiner,merger,encoder_solver,decoder_solver,refiner_solver,merger_solver)